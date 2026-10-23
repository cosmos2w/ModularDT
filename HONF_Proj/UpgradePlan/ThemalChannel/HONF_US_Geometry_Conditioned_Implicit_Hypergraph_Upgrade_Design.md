# U-S upgrade design: implicit hypergraphs with a bounded E-grid spatial prior

**Status:** revised proposal for one bounded, matched U-S adaptation; not an implementation report, a measured improvement, or authorization to modify active models, train, or run inverse optimization.

**Prepared:** 23 October 2026. The current epoch-1000 evidence is summarized from the Round11 report. Existing development checkpoints, current formal experiments, historical numerical failures, dataset memberships and physical-reference arrays remain unchanged.

**Design revision:** the user authorized retaining and sharpening this proposal. Its broader direction remains reusable query-indexed implicit hypergraphs; the selected first intervention is deliberately narrower. No implementation, training or evaluation is authorized by this document revision.

**Central idea:** test one externally scheduled, bounded spatial penalty on direct query-to-fixed-E-grid scores, with the E-grid radii fixed from the adapter's physical domain frame lengths. This is an E-grid locality prior; at fixed query, domain and radii, it adds no explicit module-translation sensitivity. Existing learned query-to-source geometry features and layout-conditioned prepared context remain active. The resulting exact-positive query-to-record sets are still describable as implicit hyperedges, but no new group bank or module-layout-conditioned selector is introduced. For a later authorized adaptation, both arms start from the literal full-TRAIN U-S epoch-1000 checkpoint and inherit its normalizer; they are a matched adaptation, not an exact resume or a new quarter-data experiment.

## 1. Executive recommendation

Develop one primary capability, provisionally named e_grid_spatial_prior_v1, by subtracting a weak, dimensionless query-to-E-grid distance penalty from the existing U-S E score. Keep the current continuous gate, guarded reduction, E-first decoder, Thermal source law, M score, M local envelope and contextual equations unchanged. This tests a spatial prior on E access; it does not add explicit dependence of the prior on module positions.

Keep learned radii, spatial regularity, calibration changes and candidate-local indexing out of the primary pair. Consider any one of them only as a separately named later amendment if measured evidence identifies a specific failure. The primary penalty uses fixed shared radii and one externally scheduled strength; neither parameter is fit from DEV outcomes.

Keep three questions separate: does the E-grid prior preserve native fields and responses; does it reduce actual executed work and complete cost; and does the adapted model give reusable, useful sensitivity or decisions in inverse design? None follows from a visually cleaner hypergraph or from the prior's spatial form.

The primary endpoint has no query-to-seed mixture, allocated group count, fixed group bank or record-to-group assignment matrix. Its existing learned object remains a field of direct record-access functions; the amendment adds only a fixed-radius E-grid prior to that field. Explicit compact candidate search is a later, separately versioned option only when profiling demonstrates that all-record scoring is a material bottleneck and the new support law can be qualified.

The [U-H companion proposal](HONF_UH_Soft_Merged_Hyperedge_Upgrade_Design.md) separately develops explicit compact groups and, conditionally, factorized contextual exchange. Neither amendment depends on implementing the other; compare their fields, executed work and complete cost without assigning an invented final group count to this direct selector.

## 2. Existing U-S mechanism and evidence

### 2.1 What is already implemented

Current U-S already performs two applied typed contextual cycles in the order MM -> EM -> EE -> ME. Route names give destination first and donor second. Its context gates control actual value exchange; they are not merely graphs plotted after an unconstrained dense encoder. The original record axes, own-record residuals, native masks and measures remain present.

Final query access is direct. Existing pre-E M/E reader logits are calibrated separately by type and passed through a continuous exact-zero gate. No final seed-directory composition is required in direct mode. Dense score construction, row calibration and information normalizers remain control-plane dependencies.

~~~mermaid
flowchart TD
    X["Original local M / E records"] --> P["Two applied typed contextual cycles"]
    P --> Z["Prepared original record states"]
    Q["Receiver query q"] --> L["Pre-E query-to-record logits"]
    Z --> L
    L --> C["Typed direct score calibration"]
    C --> B["Candidate E-only bounded grid prior"]
    B --> GE["Continuous exact-zero E access"]
    C --> GM["Existing M access and local envelope"]
    GE --> E["Individual query-conditioned E values"]
    E --> QE["E-conditioned query"]
    QE --> M["Individual M responses and local branch"]
    GM --> M
    M --> F["Native fields and signed source coefficients"]
~~~

**Figure 1. Current U-S dependency chain with the proposed E-only score term.** The same contextual architecture as other unified sparse models remains upstream. The direct selector changes how a receiver obtains final access; it does not remove all upstream attention or all record-to-record dependence. The added term uses fixed E-grid coordinates and introduces no explicit module-translation sensitivity at fixed query and domain.

### 2.2 Current direct gate

For type t, head h, query q and eligible donor i, let L be the pre-E score. The existing calibrated score is schematically:

\[
r_i^t(q)=\kappa_t\sum_h\omega_{th}
\left[L_{hi}^t(q)-\max_{j\in\mathcal A_t(q)}L_{hj}^t(q)\right]+\beta_t,
\qquad \omega_{th}\geq 0,\quad\sum_h\omega_{th}=1.
\]

Here the eligible set is denoted by \(\mathcal A_t(q)\); it is not the environment type E. Per-type scale bounds, masks and the distinction between presence and positive information measure remain explicit implementation contracts.

\[
\psi(r)=\frac{[r]_+^2}{1+[r]_+^2},
\qquad a_i^t(q)=\psi(r_i^t(q)).
\]

\[
g_i^M(q)=\ell_i(q)+(1-\ell_i(q))a_i^M(q),
\qquad g_j^E(q)=a_j^E(q).
\]

The original smooth local envelope is \(\ell_i\). Its positive support is a compulsory floor for the current work accounting; the separate local reader remains intact. No hard nearest-donor rescue is added when E access closes.

The gate is C1 at zero, but the complete score is not automatically C1: the inherited row maximum is continuous and generally nondifferentiable at changing maximizing donors. Input geometry features and physical extraction introduce further boundaries. Section 9 treats this separately rather than attributing global smoothness to the final gate alone.

### 2.3 Current physical reduction and response

For one information head and type, the inherited reduction is:

\[
w_i(q)=\frac{\mu_i\exp L_i(q)}
{\sum_{j\in\mathcal A_t(q),\,\mu_j>0}\mu_j\exp L_j(q)},\qquad
Z(q)=\sum_i g_i(q)w_i(q),
\]

\[
c(q)=\frac{\sum_i g_i(q)w_i(q)m_i(q)}
{Z(q)+\varepsilon(1-Z(q))^2},
\qquad \varepsilon=10^{-3}.
\]

An empty eligible information type returns its declared zero message. The zero-message limit, mass-scaled contextual projection bias and explicit zero-gradient paths required by the optimizer contract must remain correct.

\[
m_j^E(q)=\Psi_E(q_0,z_j^E,q-x_j),\qquad
q_E=q_0+A_Ec_E(q),\qquad
m_i^M(q)=\Psi_M(q_E,z_i^M,q-x_i).
\]

Pre-E access uses q0; final M information scores and local/field responses use qE. Do not make access depend on E context that can only be computed after that access is chosen.

For the current Thermal capability:

\[
K_i(q;X)=g_i^M(q;X)\,k_i(q_E,z_i^M,q-x_i)
+K_i^{\mathrm{local}}(q_E,X,q),
\qquad
T(q;X,h)=\sum_iK_i(q;X)h_i.
\]

E records are not heaters. Information-attention weights and quadrature are not multiplied into the signed source sum a second time. A present zero-information-measure module can still be a physical heat source and must not lose its source column.

### 2.4 Evidence motivating this proposal

The current Round11 report evaluates literal epoch-1000 full-TRAIN U-D/H/S checkpoints alongside classic 1404/1804 on the same exposed canonical VAL89 receivers. U-S has the lowest equal-case mean error on seven of nine measured quantities within the U triplet, while U-H leads fluid temperature and vorticity; classic 1804 retains the transverse-velocity advantage and some temperature-tail advantages. Training objectives, normalization and source protocol differ across the classic comparison, so these measurements motivate a bounded U-S follow-up but do not establish an organizer-only cause or universal model ranking.

The epoch-1000 organizer panel contains four input-selected VAL89 cases plus two previously frozen TRAIN24 endpoints, with eight common query indices per case. Across the 89-case final-reader board, U-S has exact-positive support on 48.71% of eligible M rows and 44.46% of eligible E rows; its configured dense executor nevertheless executes 100% of final M and E value rows. The measured support therefore establishes a selective access field, not reduced final value work or complete-call acceleration.

No epoch-1000 response/candidate replay or optional E-access intervention was run. The new checkpoint's improved field means do not establish useful geometry-design gradients, moved-layout transfer or inverse validity. The older e5000 and saved-candidate results remain historical evidence for their own checkpoints and panels; they are not the primary basis for this proposed adaptation.

## 3. The computed object: a query-indexed implicit hypergraph

For a fixed scene X, define the weighted membership field:

\[
\mathcal G_\theta(X):q\mapsto
\big(g_1(q;X),\ldots,g_{N(X)}(q;X)\big).
\]

Each query induces one overlapping hyperedge:

\[
\mathcal H_q(X)=\{i:g_i(q;X)>0\}.
\]

The model shares parameters and contextual encoders across queries; it does not store an independent trainable membership vector for every spatial point. Equal or similar inputs can produce related memberships through the existing learned rule, now perturbed for E by the fixed-coordinate prior. Repeated co-retrieval is an emergent organization, not a required factorization through a fixed set of groups. The added distance term compares q with fixed E-grid coordinates, so it has no explicit module-translation derivative at fixed q, domain and radii; existing learned geometry scores and layout-conditioned prepared context remain separate live paths.

~~~mermaid
flowchart LR
    RULE["Existing direct rule plus fixed E-grid prior"]
    RULE --> QA["Query A"]
    RULE --> QB["Query B"]
    RULE --> QC["Query C"]
    QA --> HA["Implicit edge A<br/>M1, M2, E1, E2"]
    QB --> HB["Implicit edge B<br/>M2, E2, E3"]
    QC --> HC["Implicit edge C<br/>M3, E3, E4"]
    HA --> V["Original record values<br/>No pooled hyperedge value"]
    HB --> V
    HC --> V
~~~

**Figure 2. Query-indexed overlapping hyperedges.** This is an illustrative schematic, not a learned result. M2/E2 and E3 participate in multiple edges. The membership rule is reusable; edge identity is tied to the query and scene rather than a permanent stored group slot.

Calling this an implicit hypergraph describes the computed sets. It does not prove an irreducible higher-order physical interaction, superiority to a graph message-passing implementation, or a unique recovered physical partition.

## 4. Scope, preserved semantics and independent model identity

This note defines a possible bounded matched adaptation but authorizes no implementation, fit, solver call, inverse search, process interruption or storage migration. Any later execution requires its own explicit run authorization, resource budget, output budget and stop horizon.

Preserve current formal workers and scientific checkpoints. Any future implementation that might affect imported code uses a separate ordinary worktree or isolated capability; do not patch the source beneath active workers. Historical numerical failures and original result arrays remain unchanged.

The first candidate changes final E selection only. Keep contextual equations, M access and its local envelope, physical source contraction, query counts, native losses, units, training membership and inherited normalization unchanged. A later module-layout feature or geometry-aware contextual selector is an additional scientific change and is conditional, not part of the first candidate.

No per-case trainable routing table, numeric native-ID embedding, target-conditioned geometry feature, hidden global-summary correction, dense partner predictor or parent-prediction supervision is added. Original IDs label correspondence and intervention targets, not learned importance.

## 5. Design requirements and non-goals

| Requirement | Intended observable | Insufficient substitute |
|---|---|---|
| Useful sparsity | Fewer exact-positive selected rows and, separately, fewer actually executed value rows at comparable fields | Lower gate sum or prettier support plot |
| E-grid prior | Measured response to q-to-fixed-E-grid distance in the declared normalized frame | A claim that the prior depends on module translation |
| Query consistency | Related access across nearby legal receivers | Constant access for every query |
| Module-layout sensitivity | Full-model live derivative through existing context and local-feature paths, assessed separately from the new prior | Attributing a module-translation derivative to the E-grid penalty |
| Inverse reuse | Valid response-factor reuse or transferable live selection rule | Reusing stale prepared tensors after geometry changes |
| Practical execution | Measured complete cost with all control work charged | Counting logical omissions as time saved |

The proposal does not require a fixed number of groups, identical supports across layouts, all sources being local, universal rotational invariance, or an unconditional sparse speedup. Its new prior alone is not expected to respond to module translation when q, domain and fixed radii are held constant.

## 6. Core upgrade: a fixed-E-grid spatial prior

### 6.1 Geometry features and frame semantics

For the primary Thermal candidate, use only the query coordinate q, each original fixed E-grid coordinate x_j^E, the adapter's normalized physical frame and its prescribed flow direction when available. Do not use module positions, true fields, reference errors, candidate rankings or unavailable physical metadata in the new prior. Existing contextual states and learned query-source scores continue to use their current legitimate inputs.

For Thermal with a prescribed flow direction e_parallel, define:

\[
\Delta x_j^E=q-x_j^E,\qquad
D_j^E=
\frac{(\Delta x_j^E\cdot e_\parallel)^2}{r_{\parallel,E}^2}
+\frac{\|\Delta x_j^E-(\Delta x_j^E\cdot e_\parallel)e_\parallel\|^2}
{r_{\perp,E}^2}.
\]

Use one shared fixed radius pair for every E record, with \(r_{\parallel,E}=L_\parallel\) and \(r_{\perp,E}=L_\perp\), where L values are the adapter's physical frame lengths in the same native coordinate units as q and the E coordinates. The dimensionless radius multipliers are both1.0 and fixed; domain rescaling transforms the physical radii with the coordinates. These radii are finite, positive and frozen by the candidate's frame recipe; they define a reproducible baseline and are not learned or claimed optimal. D is dimensionless. This choice isolates the score bias from radius fitting and avoids a new per-query geometry network.

For a case without a meaningful external direction, use the isotropic normalized-frame metric \(D_j^E=\|q-x_j^E\|^2/L^2\) with the adapter's declared scale L rather than inventing a direction from record order. Squared longitudinal distance is upstream/downstream symmetric; legitimate signed flow-relative features can remain in the existing content score. An asymmetric geometric penalty is a later explicit option, not a built-in wake formula.

Permuting E records and their coordinates together must permute the penalty and gates identically. Any frame transformation must transform queries, E coordinates, boundaries and vector operating conditions together. The existing model is not thereby made rotation-equivariant; a fixed channel with a fixed inlet is not invariant under rotating modules alone. Reflection symmetry is tested only when materials and boundary conditions actually share it.

Because x_j^E is a fixed grid in the current Thermal adapter, translating modules while holding q, the domain and radii fixed leaves D_j^E unchanged exactly. This statement is about the added prior only. Existing query-source geometry features and the applied contextual preparation can still carry module-layout dependence into the full model.

### 6.2 Minimal score amendment

Use the current calibrated E direct score as the baseline and leave M access exactly unchanged:

\[
\widetilde r_j^E(q)
=r_j^{E,\mathrm{US}}(q)-\lambda_E(u)\rho(D_j^E(q)),
\qquad
\rho(D)=\frac{D}{1+D},
\qquad \lambda_E(u)\geq0.
\]

\[
\widetilde a_j^E(q)=\psi(\widetilde r_j^E(q)),\qquad g_i^M(q)=g_i^{M,\mathrm{US}}(q).
\]

The bounded distance penalty expresses a weak preference for E records near the receiver in the normalized E-grid frame while allowing strong content evidence to retain distant E donors. Unlike an unbounded distance penalty or finite support radius, it does not structurally forbid long-range influence. The penalty is in the existing calibrated score units; its output is not a probability or a physical distance.

Use one externally scheduled E-strength scalar. The selected cap rule is the 25th percentile, with linear interpolation, of positive calibrated E-score margins from the literal parent checkpoint on the existing frozen input-selected TRAIN24 panel, using1024 legal primary queries per case from one declared case-epoch sampling draw. Save the exact case IDs, input/geometry identity, query draw and positive-margin summary; selecting this panel uses no target fields, and margin computation is a model diagnostic rather than a physical label. If there are no positive margins, set the cap to zero and declare the candidate non-informative rather than tuning on DEV. Ramp linearly from zero to this cap over the first 10% of the declared adaptation optimizer-update horizon, then hold it fixed. This percentile and ramp are practical proposal defaults, not measured optima or a guaranteed closure fraction; the bounded distance multiplier can leave many gates positive. Use one rule, not a strength grid or a learned dual. The score-calibration function is unchanged, its learned parameters remain trainable in both arms, and the fixed radii receive no gradients.

At \(\lambda_E=0\), with unchanged score calibration and parameters, this gate recovers current direct access exactly. The scheduled scalar is part of the child-run identity and its update position is checkpointed. For the proposed 100-epoch pair there are 1300 optimizer updates and a130-update ramp: at update cursor u, lambda_E=cap*min(u/130,1), shared by all microbatches and query tiles of that update. Validation and exported inference use the lambda saved at that checkpoint's cursor, so an early checkpoint is not silently evaluated with a different final-cap function. The completed child uses the fixed cap thereafter. The E-grid prior remains deterministic across query tiles and batch composition.

### 6.3 Why this can sparsify, and why it may fail

A weakly positive score can cross zero after the penalty, causing exact omission. A strongly useful distant record can stay open. If all scores are far from zero, the change may attenuate coefficients without reducing any executed rows. Report both outcomes.

The calibrated score and externally chosen strength can compensate for one another; neither the cap nor the fixed frame-scale radii identify a physical length or importance. Compare the actual access field and native response, not the config label alone.

Closed C1 gates have zero local derivative. An omitted edge is not guaranteed to reopen just because a dense executor evaluates its value. Near-open initialization, a gradual ramp and shared parameter updates are limited mitigations; explicitly test recovery in a nondegenerate fixture and record failure rather than promise an unbiased restoration mechanism.

## 7. Sparse work control and protection against misleading compression

Retain one coherent M/E/context work controller. Current code already combines observed exact support for feedback with the differentiable surrogate:

\[
C_t=\sum_{q,i\in t}\mathbf 1[g_i(q)>0],
\qquad
\widetilde C_t=\sum_{q,i\in t}\eta_\delta(g_i(q)),
\qquad
\eta_\delta(g)=\frac{(1+\delta)g}{g+\delta}.
\]

The proxy is not an exact expected count. Preserve case/request accumulation and equal-case macro normalization, along with compulsory local floors. Do not replace a macro-level penalty with a mean of independently hinged microbatch penalties.

For the first geometry experiment, keep the parent's work targets and controller treatment matched to its unchanged U-S control. This isolates geometry conditioning from stronger pruning pressure. Only a separately budgeted comparison may alter target work.

Use three distinct counts in every comparison. **Scored rows** are eligible E query-record pairs whose calibrated score and normalizer are evaluated; Level A retains all of them. **Selected rows** are exact-positive gates \(g_j^E(q)>0\), counted against eligible rows with no display threshold. **Executed value rows** are the rows actually passed through the E value network by the chosen backend; a dense-weighted backend executes its full rectangle even when gates are zero, while a packed backend reports its gathered rows. Never infer executed work from selected support.

Log actual positive rows, proxy rows, eligible rows, local floors, type/role strata and positive gates below the existing magnitude bins. Report score construction, selected support and executed rows separately, then measure complete cost with a named timing boundary. In a dense-weighted backend, executed rectangular values are counted separately even when their logical gates are zero.

Inspect attenuation compensation: a declining gate can coexist with growing ungated messages or source responses. Record E-context mass, qE-q0, signed K and native field/coordinate sensitivity. Gate shrinkage alone is neither lower cost nor better inverse conditioning.

Do not penalize all source responses toward zero to make the model appear stable. Any new response-amplitude regularizer requires native-unit normalization and its own motivation; it is not part of the first candidate.

## 8. Spatial regularity without forcing a fixed partition

This section describes a possible later refinement only. It is disabled in both arms of the primary matched adaptation; do not add it alongside the E-grid prior.

### 8.1 Train-time neighborhood, independent inference

Build a small input-only neighborhood of receiver pairs in the same legal physical role and connected material/fluid region. The neighborhood is used for a training regularizer only; ordinary prediction of a query cannot depend on which other queries happen to be supplied in the same batch.

Use original native coordinates or valid continuous-core queries. Exclude cross-solid/interface pairs unless the expected regularity is explicitly justified. Coincident duplicate queries are handled separately, not divided by a vanishing distance.

For dimensionless coordinates \(\bar q\), define an optional excess-variation penalty by type:

\[
\mathcal R_q=
\sum_{t\in\{M,E\}}\omega_t
\mathbb E_{(q,q')}
\left[
\frac{\|g^t(q)-g^t(q')\|_2}{\sqrt{\max(1,N_t)}}
-L_t\|\bar q-\bar q'\|_2
\right]_+^2.
\]

The declared L values are regularization scales, not measured physical Lipschitz constants. Select them once from an input-selected TRAIN diagnostic of the baseline and report that heuristic. Do not retune them against favorable DEV errors.

### 8.2 Meaning and limits

This allows moderate, genuine variation and discourages only changes above the chosen scale. It is less restrictive than requiring identical binary supports or globally identical gates. Keep M/E normalization separate so the larger E catalogue does not dominate.

A smooth gate can still induce support changes at zero; that is allowed. Use coefficient variation and field derivatives as the primary stability measurements, with support changes as an additional count. A tiny change in a gate can change a binary mask without producing a significant field jump.

Oversmoothing is a real failure mode, especially near material surfaces, thermal peaks and boundary layers. If native fields or useful sensitivities deteriorate, weaken or reject this amendment. Do not restore accuracy through an unrecorded dense correction.

The first implementation should not add the E-grid prior and spatial regularization together. Consider this regularizer only as a separately scoped later amendment if the measured access field is fragmented or excessively steep.

## 9. Derivative quality: inherited kinks and legal design perturbations

### 9.1 The current row maximum

The inherited gate preactivation subtracts an eligible row maximum. At a tie, changing the maximizing donor can change its derivative even while the resulting gate remains positive. C1 psi does not remove that kink. Stable log-sum-exp used inside ordinary softmax is algebraically different: its numerical maximum shift cancels from the mathematical softmax, whereas this maximum is part of the declared gate function.

First retain the existing calibration and measure actual tie neighborhoods separately from gate-zero transitions. If a diagnosed calibration kink materially harms design derivatives, one optional new calibration is:

\[
c_h(q)=\tau_c\log\left(
\frac{\sum_{j\in\mathcal A_t(q)}\nu_j
\exp(L_{hj}(q)/\tau_c)}
{\sum_{j\in\mathcal A_t(q)}\nu_j}
\right),
\qquad
r_i^t=\kappa_t\sum_h\omega_{th}(L_{hi}-c_h)+\beta_t.
\]

Here tau_c is positive and fixed for a candidate. The positive control weights nu are declared separately from physical measures; uniform eligible-record weights are a simple initial choice. Empty sets use the explicit zero-type convention. This smooth log-mean-exp calibration changes the model, has cardinality/weight semantics, and does not exactly recover a finite-temperature row-max parent.

Test this only if needed, as a separate capability or ablation. It does not remove kinks already present in contextual calibration or raw geometric features. A whole-model C1 claim would require auditing those paths too.

### 9.2 Layout changes require live rebuilding

Let d denote continuous design variables and X(d) the resulting legitimate geometry. Recompute all changed local inputs, geometry features, contextual states, direct scores, gates and native response extraction. Keep domain/frame fixed when that is the intended design problem; preserve original module correspondence.

Check full autodiff against represented finite differences along a few legal translations or size changes away from raster/mask boundaries. At physical interfaces or topology changes, label the separate derivative scope rather than forcing an inappropriate smoothness test.

An optional design-neighborhood excess-variation penalty can use corresponding records and receivers at d and d+delta. Normalize design coordinates by declared physical scales. It must permit real sensitivity rather than penalizing every change; deploy only after a named instability is observed. It supplies a model-side stability prior, not physical target derivatives.

No perturbed layout is supervised against the original layout's target. No output at an unsolved perturbation is relabeled physical truth.

## 10. Training objective and collapse controls

The proposed primary training structure is the parent's unchanged objective and work loss plus the E-grid score term; set both optional regularizer weights to zero:

\[
\mathcal L=
\mathcal L_{\mathrm{native}}
+\mathcal L_{\mathrm{existing\ typed\ work}}
+\lambda_q\mathcal R_q
+\lambda_d\mathcal R_d.
\]

For the primary pair, \(\lambda_q=\lambda_d=0\); the scheduled E-grid bias is applied inside the direct score. Both optional regularizers remain disabled; activate one only in a later amendment justified by diagnosis. Do not add a large collection of entropy, diversity, balance and clustering losses merely because the selection can be called a hypergraph.

Retain all native field/auxiliary/response objectives of the explicitly selected recipe. Temperature, flow, q-proxy, surface/material roles and source peaks remain separately reported. An improving pooled loss does not compensate for a severe role or stratum failure.

An L1 penalty on normalized attention weights is constant and cannot encourage sparsity. Entropy reduction can concentrate mass while leaving every access positive. L1 on unnormalized gates can encourage attenuation or a large number of tiny positive reads. Use the existing exact-support feedback and demonstrate actual execution.

Do not enforce equal numbers of selected M/E records or equal usage of every record. An original physical source can be rare but essential for a peak or constraint. A completely closed E branch is a legitimate diagnostic outcome, not automatically a useful environmental organizer.

Keep native gradients through contextual preparation, direct gates, E conditioning and source responses. Do not detach geometry or routing to manufacture faster or apparently stable training.

## 11. What inverse design can reuse

### 11.1 Fixed layout, changing heat allocation

At fixed X and fixed receiver set, current Thermal has:

\[
\mathbf T=K(X)\mathbf h.
\]

Compute native source coefficients once and reuse them for many admissible heat vectors. Preserve original source-column IDs, native temperature scaling, local contributions, surface/material extraction and peak recomputation. This reuse follows from the existing Thermal source law and can be qualified independently of the organizer amendment; it is not evidence that the new prior discovered better inverse physics. Gate selection and the saved checkpoint's lambda_E are heat-independent in this capability; the completed child uses the fixed cap.

For example, a fixed-layout surrogate minimax allocation can solve:

\[
\min_{h,t}\ t,\qquad
K_{\mathrm{receivers}}h\leq t\mathbf1,\quad
h_{\min}\leq h\leq h_{\max},\quad
\mathbf1^\top h=H_{\mathrm{total}},
\]

when this is the user's requested receiver objective and the constraints are supplied. It is not automatically identical to the existing saved-candidate smooth-peak objective. A different objective must be labeled, not used to improve an old ranking retrospectively.

Pressure is heat-independent here. An infeasible layout receives no claim of pressure repair through heat allocation. A surrogate-optimal heat vector remains unverified without compatible physical references.

### 11.2 Fixed layout, changing queries

Reuse prepared original-record states, static projections and legal geometry metadata. Recompute query-specific scores, E-grid distances, access, individual values and native readout. Cache exact repeated queries only with model/scene/query identity intact. Do not reuse one query's coefficient vector at another query because their support happens to match.

### 11.3 Changing geometry or prescribed operating conditions

Reuse learned parameters and the organization-generation rule, not old numerical gates or stale prepared tensors. Even though fixed E-grid coordinates make the new distance term itself invariant to moving modules at fixed q and domain, the full U-S score/context and response may change with module geometry and must be rebuilt. The updated model is:

\[
J(d)=\mathcal J\!\left(F_\theta(X(d),h(d)),d\right),
\qquad
\frac{dJ}{dd}
=\frac{\partial\mathcal J}{\partial F}
\frac{dF_\theta}{dd}
+\frac{\partial\mathcal J}{\partial d}.
\]

The total derivative includes geometry features, contextual states, all live calibration/normalization paths, direct gates, E conditioning, values and the applicable native extraction. A derivative through only the selected value network is a different fixed-plan diagnostic.

~~~mermaid
flowchart LR
    D["Design variables"] --> X["Rebuild legitimate geometry"]
    X --> P["Prepare original record states"]
    P --> G["Apply the same learned selection rule"]
    G --> V["Recompute original values and response K"]
    V --> J["Objective and constraints"]
    J --> U["Full live-model design gradient"]
    U --> D
~~~

**Figure 3. Reuse under geometry changes.** The neural rule is reused while its geometry-dependent outputs are rebuilt. This diagram is a proposed differentiable surrogate workflow, not a validated inverse optimization trail or permission to launch one.

### 11.4 Reuse table

| Change | Reusable objects | Objects requiring fresh computation |
|---|---|---|
| Heat h only, same layout/queries | Prepared scene, access, K and fixed extraction maps | Temperature contraction and requested objective |
| New receiver q, same scene | Original contextual states, static projections and fixed E coordinates | Query scores/E-grid distances/gates/values and readout |
| Module positions/sizes | Model parameters and rule | Geometry, context, access, K and affected masks |
| Operating condition/material input | Shared parameters within tested capability | All affected features, states and fields |
| Model weights or normalization | Nothing numerical from the old prepared graph by default | Prepared states and all dependent factors |
| New topology or record inventory | Shared variable-count rule, if qualified | Inventory, preparation, eligibility and derivative scope |

## 12. Local design neighborhoods and honest approximate reuse

Exact K reuse applies only when K's dependencies are unchanged. For geometry optimization, one may later use a trust-region approximation:

\[
K(d_0+\Delta d)\approx
K(d_0)+J_K(d_0)\Delta d.
\]

This is a local surrogate of the already learned model, not a new physical solution. Bound its application to a declared local region and check selected trial points with a fresh full model evaluation. If the approximation error or constraint margin is unacceptable, rebuild/relinearize or reject that proposed step.

Support changes do not automatically invalidate the first derivative of a C1 gate, but finite-step approximation can still fail near steep transitions, calibration ties or physical boundaries. A frozen index list is permitted only in a labeled local test whose omitted contributions/derivatives have been assessed; it is not the definition of full live geometry inference.

Do not store differentiable graphs across optimizer steps or allow previous prepared tensors to survive changed geometry silently. Cache keys should express ordinary semantic identity; no new campaign admission service is needed.

The first upgrade need not implement trust-region acceleration. Its required inverse artifact is a tested live derivative and fixed-layout heat reuse interface, plus bounded saved-reference decision diagnostics.

## 13. Efficiency levels and their distinct claims

### 13.1 Level A: E-grid-prior selection with the current score scope

The primary E-grid-prior gate keeps all-eligible scores, current row calibration and the original reader denominator. Score construction remains O(QN), plus an O(QN_E) normalized distance term; U contextual preparation still forms dense pair scores of order O(N squared). It can reduce wide nonlinear value work only if new exact zeros appear and the selected backend actually omits those value rows; it does not reduce all organizational work.

The outer query-tile owner computes q-by-E distances in the same query tile as the E scores, fuses the penalty into that tile, and releases the temporary before the next tile; do not materialize a scene-wide Q-by-E distance array. Reuse fixed E coordinates and static record projections per preparation and query projections per tile; aggregate counters rather than synchronizing every tile. Use existing segment or dense-weighted execution on the same function. Complete cost includes preparation, score and distance construction, sorting/packing, gathers, backward recomputation, work derivatives, optimizer, transfers and output within explicitly named scopes.

### 13.2 Level B: optional compact geometric footprints

Only if score construction remains a material bottleneck and nonlocal response tolerates it, introduce a new finite-footprint E capability:

\[
g_j^E(q)=\kappa(D_j^E(q))\,\psi(\widetilde r_j^E(q)),
\qquad
\kappa(D)=[1-D]_+^2.
\]

At D_j^E=1, both the footprint value and its first derivative vanish. This permits conservative spatial broad-phase enumeration of all D_j^E<1 records without introducing a nonzero value jump at footprint birth/death, assuming other constituent functions are sufficiently regular. Boundary derivatives and exact eligibility still require tests.

A finite footprint can remove real long-range environmental influence. It is a separate scientific restriction, has no general exact equality to the existing model at finite radius, and is not the default merely because it enables an index.

Crucially, skipping distant scores is not same-function under the inherited row-max and all-eligible softmax denominator. Even a donor with zero final gate can influence those control quantities. Keeping the old denominator preserves the O(QN) residual. Dropping it silently would change the model.

### 13.3 A possible fully candidate-local operator, only as a separate experiment

If a score-local design is justified, specify the new information law explicitly, for one type/head:

\[
v_j^E(q)=
\frac{\mu_j}{\mu_{\mathrm{ref},E}}\,
\kappa(D_j^E(q))\,
\psi(s^{\mathrm{local}}_j(q))\,
\exp(\bar L_j(q)),
\qquad
c_E(q)=\frac{\sum_j v_j^E(q)m_j^E(q)}
{\zeta_E+\sum_j v_j^E(q)},
\qquad \zeta_E>0.
\]

The local score cannot require a maximum over excluded records. A bounded logit, for example \(\bar L=L_0\tanh(L/L_0)\), makes overflow control independent of the changing candidate set. mu_ref is a fixed adapter measure scale and zeta is dimensionless. Both are declared model settings. Missing mass belongs to a fixed zero message; mass-scaled projection bias remains explicit.

Only candidates with positive footprint need their scores and values. As a candidate enters at D=1, v and its first derivative vanish under the stated regularity assumptions. Positive-measure and empty-type handling remain explicit. Physically present zero-measure M sources are outside this E-only amendment.

This changes calibration, information normalization and attenuation together; it is not a faster implementation of Level A or the old guarded function. It need not recover an all-open old parent. Qualify its own literal equations, then separately assess physical deviation and adaptation. Do not launch it together with multiple new gate losses.

### 13.4 Complexity accounting

| Scope | Current / primary Level A | Optional candidate-local E operator |
|---|---|---|
| Applied contextual pair scores | O(N squared), unchanged | O(N squared), unchanged |
| Decoder query-record scores | O(QN) | O(sum of candidate counts) for amended E branch |
| Final seed-directory composition | Absent in direct mode | Absent |
| Decoder nonlinear values | Actual positive rows or dense rectangle, backend-dependent | Positive candidate rows, backend-dependent |
| Spatial index | Not required | Construction, update and enumeration charged |
| M physical source columns | Original M | Original M |

For illustration, Q=8192 and E=192 require 1,572,864 E score pairs under a full query-record score pass. A candidate average of 24 would require 196,608 pairs for an actually candidate-local branch, an eightfold reduction in that isolated count. This assumes the new branch and index work; it is not an end-to-end speed estimate or a recommended retention target.

Do not claim reduced upstream complexity from a decoder-only footprint. Geometry-dependent sparse contextual scoring would be an additional future model, requiring its own stage semantics and physical comparison.

## 14. Training and deployment must use compatible selection

Primary training and inference use the same continuous direct function. No post-training top-K, magnitude cutoff, inference-only radius reduction or target-dependent selection is permitted. A plotted cutoff may aid visualization only if the exact-positive counts remain separately shown.

With finite footprints, training may use conservative buffered candidate lists, but must recompute live coefficients and include every potentially active candidate at the current geometry. Refresh list topology before the buffer guarantee fails. A buffer is an execution superset, not permission to discard live derivatives or to charge only positive values when all buffered values were computed.

Automatic differentiation through discrete spatial-index construction is not itself the desired derivative. The selected analytical function can have a correct sparse derivative when contributions and first derivatives vanish at exclusion boundaries and candidate completeness is guaranteed. Validate this against a full-record literal reference, including record entry and exit.

The current row-maximum score and all-eligible denominator cannot be dropped from such a reference. If the new candidate-local law is used, name that function explicitly in configuration and exports.

## 15. Interpreting and measuring the implicit organization

Do not require hard clustering to make the implicit hypergraph visible. On a fixed low/small/high-M panel, show:

- Per-record spatial gate footprints in the original physical frame.
- A query-by-record coefficient matrix and exact-positive mask with original IDs.
- Backend-specific executed M/E value rows, information mass and tiny-positive distributions.
- Overlap between neighboring query hyperedges, coefficient variation and role/stratum differences.
- How the full model responds to a legal module-layout perturbation, while separately confirming that the new fixed-E-grid distance term itself is unchanged at fixed q and domain.

A diagnostic co-retrieval matrix can be estimated on a bounded panel:

\[
R_{ij}=
\frac{1}{|\mathcal Q_{\mathrm{panel}}|}
\sum_{q\in\mathcal Q_{\mathrm{panel}}}
\mathbf1[g_i(q)>0]\mathbf1[g_j(q)>0].
\]

This is a query-distribution-dependent descriptive statistic. Compute it only on the small requested panel or sampled record pairs, not as an O(QN squared) training loss. Similar co-retrieval does not establish a unique group or causal coupling.

Optional clustering of diagnostic footprints is for interpretation/cache exploration only. Feeding those clusters back into execution creates a new organizational constraint and requires a separately stated experiment. The primary model remains implicit.

## 16. Worked example: selective geometry bias without forced locality

Consider M1/M2/M3 and E1/E2/E3/E4. The following E scores are synthetic dimensionless numbers for one query, not a trained model or physical data. Take \(\lambda_E=0.4\), \(\rho(D)=D/(1+D)\), and the current psi gate:

| Quantity | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
| Current direct preactivation r | 0.30 | 0.05 | -0.20 | 0.90 |
| Dimensionless distance D | 0 | 1 | 4 | 0.5 |
| Geometric penalty | 0 | 0.20 | 0.32 | 0.133333 |
| Amended preactivation | 0.30 | -0.15 | -0.52 | 0.766667 |
| Original positive access | Yes | Yes | No | Yes |
| Amended positive access | Yes | No | No | Yes |

E2 closes, while E4 remains active despite a nonzero distance penalty because its content score is strong. The example reduces exact-positive selected E rows from three to two; a dense executor may still evaluate all four E values. It proves only the gate arithmetic, not improved prediction, complete timing or the importance of any E record.

~~~mermaid
flowchart LR
    Q["Same receiver query"] --> OLD["Content-only direct selection<br/>E1, E2, E4"]
    Q --> NEW["Content plus weak geometry bias<br/>E1, E4"]
    OLD --> O["Original query-conditioned values"]
    NEW --> N["Original query-conditioned values"]
    N --> P["Recompute E context and M response<br/>Measure fields before calling this useful"]
~~~

**Figure 4. Synthetic direct-support change.** No seed group is selected or merged. The exact-positive selected hyperedge changes because the shared record-selection function changes; execution depends on the chosen backend. The numerical table does not promise this closure will be harmless.

## 17. Proposed interfaces and execution ownership

Keep the maintained prepare/decode boundary. Model capability, fixed radius scales, prior cap/ramp and normalization identity define scientific state. The child checkpoint must save the parent checkpoint hash, inherited normalizer identity, E-frame/scales, fixed radii, cap rule/value, schedule rule and current local update position alongside weights, optimizer, RNG and sampler state. A child-state resume and any inference export restore the same checkpoint lambda position; only the completed adaptation necessarily uses the final cap. Tile size, query packing and supported same-function backends define execution settings. The query-tile owner computes the current tile's q-by-E distance and score, then releases the temporary before advancing; no persistent all-query distance matrix is stored.

~~~python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class DirectSelectionOptions:
    # Proposed configuration sketch, not an existing implemented capability.
    geometry_mode: Literal["off", "e_grid_bounded_bias"] = "off"
    e_grid_radius_scale: tuple[float, float] = (1.0, 1.0)
    e_grid_strength_cap: float = 0.0
    e_grid_ramp_fraction: float = 0.10

def biased_e_grid_gate(raw_direct_score, dimensionless_distance, strength,
                       positive_gate):
    # Nonnegative strength and distance are validated by the public wrapper.
    bounded_distance = dimensionless_distance / (1 + dimensionless_distance)
    return positive_gate(raw_direct_score - strength * bounded_distance)

~~~

The fixed radius scales in this sketch multiply adapter-declared normalized frame scales; the strength cap is computed once by the input-only parent-margin rule above. These sketches omit masks, physical source eligibility, all-empty cases, multihead normalization, numerical range handling, optimizer state and source-law integration. They are not a training implementation. No new execution backend is required just to add the primary bias. Smooth calibration and spatial regularizers are separate later options and are not enabled by these settings.

~~~python
# Proposed usage; these are interface intentions, not executable launch code.
prepared = model.prepare(scene)
prediction = model.decode(prepared, native_queries)
description = model.describe_selection(prepared, query_ids=fixed_query_ids)

# Same geometry and exact fixed queries:
K = model.source_response(prepared, native_queries)
temperature = K @ heat_vector

# Geometry changed: rebuild rather than reusing prepared or K.
new_prepared = model.prepare(updated_scene)
new_prediction = model.decode(new_prepared, native_queries)
~~~

Ordinary caller code must not manually detach routing tensors, rebuild masks from plots, or select stale native-ID lists. Diagnostic traces are bounded and opt-in. Prepared state is invalid after scene/model/normalization changes or backward consumption according to the maintained lifetime contract.

## 18. Repository implementation map

| Existing area | Proposed responsibility | Boundary to preserve |
|---|---|---|
| continuous_joint/routing.py | Optional bounded E-grid bias in direct mode | Fixed E coordinates, fixed radii, unchanged M path and exact zero-strength recovery |
| joint_native/routing.py | Retain legacy calibration; add an explicitly selected new calibrator only if justified | Do not silently amend historical models |
| continuous_joint/primitives.py | Gate, guarded mass and independent literal helpers | New candidate-local normalization has its own capability |
| continuous_joint/budget.py | Existing actual/proxy typed work and local floors | No second conflicting sparsity controller |
| continuous_joint/unified.py and preparation.py | Existing applied cycles and state ownership | No reinstated global summary or hidden value broadcast |
| continuous_joint/reader.py | Original E-first values, post-E M attention and source law | Logical access distinct from dense executed rows |
| Thermal adapter / local input capability | Relative geometry, scales, boundaries and legitimate directions | No target-derived or heat-dependent route features |
| Maintained research runtime | Matched-adaptation config and ordinary save/resume for external lambda schedule | Persist schedule/cursor and exact parent lineage; no second engine or auto-launch |
| Query-tile execution owner | Compute and fuse q-by-E distance with the existing tile's E score | No scene-wide Q-by-E tensor or tile-dependent lambda |
| A small optional spatial-support helper | Conservative candidate enumeration for the separate footprint capability | No inaccurate nearest-K replacement |

Introduce helpers only when shared ownership is clear. Do not redesign all case adapters or implement a generic future-data platform for this Thermal experiment.

## 19. Independent numerical and semantic tests

### 19.1 Primary E-grid-prior selector

- Zero-strength recovery of unchanged direct access with identical calibration, masks and local envelope.
- Independent FP64 gate/reduction/source-sum values and VJP/JVP away from known nonsmooth boundaries.
- Exact zero crossing, finite large inputs, empty information type, zero-measure physical M source and invalid padding.
- Source permutation, query-alone/chunked/mixed-batch equality and duplicate receiver occurrences.
- Permuting E records with their coordinates permutes penalties and gates identically; query chunking and batch composition do not change the prior.
- At fixed q, domain and radii, a module translation has exactly zero derivative through the new distance term; test any full-model derivative separately through the existing context and score paths.
- Fixed frame-scale radii, the external lambda schedule and its saved/resumed cursor yield the same per-query scores independent of tile boundaries.
- Coordinate-unit rescaling leaves D unchanged when the adapter's declared scales are transformed with the coordinates; do not claim arbitrary rotation equivariance.
- First-divergence inspection through preactivation, gates, information weights, E context, qE, M response, fields, losses and gradients.

### 19.2 Derivatives and physical capability

- Gate-zero transition versus legacy row-max tie versus genuine physical interface, tested separately.
- Smooth-calibration formula against an independent log-mean-exp reference if that option is introduced.
- Legal geometry changes rebuild all dependencies; full live autodiff and represented finite differences are compared per field/unit.
- Zero heat, heat scaling and superposition, unchanged signed source-column IDs and heat-independent flow/routing.
- Surface/material extraction, curl, q-proxy and recomputed module peaks remain native.
- Empty-support zero-gradient presence and AdamW behavior stay correct; no detachment shortcut.
- Fixed-plan model-reliance tests remain distinct from live-control and physical-reference interventions.

### 19.3 Optional candidate-local execution

- Full-record literal finite-footprint function versus indexed execution, forward and derivatives, including entry/exit at D=1.
- Conservative index completeness, candidate duplication removal, variable inventory and changing learned radii.
- No hidden all-record row maximum, softmax denominator or uncharged fallback in the claimed candidate-local branch.
- Actual score/value counters and memory allocations follow candidate counts, including buffer/padding overhead.
- Query permutations, same-case repeats, empty candidate types and contextual dependence remain explicit.

Numerical tolerances are chosen before measurement using the maintained native policy, not loosened after failures. Existing source-order/segment discrepancies remain historical negatives. A failed optional backend does not invalidate unrelated diagnosis through a known reference implementation.

## 20. Bounded development and comparison sequence

Every step below is conditional on separate authorization; this proposal revision does not implement the selector, start an adaptation or launch any evaluation.

1. **Read the exact parent:** qualify the literal `thermal_u_s_full5000_v1/checkpoints/epoch_1000_model.pt` checkpoint, its hash, full-TRAIN600 membership, saved A2-D normalizer, query recipe, objective and work-controller state. This is the current full-TRAIN U-S parent, not an old e5000 U10 endpoint or an H/D parent.
2. **If design work is separately authorized, freeze the prior config without fitting:** fix the adapter frame scales as the shared E radii, compute the proposed positive-margin percentile cap from a frozen input-only TRAIN query sample, and freeze the linear ramp over the first 10% of the adaptation update horizon. If the cap is zero, stop the candidate as non-informative; do not choose a replacement from DEV.
3. **If implementation is separately authorized, qualify it:** prove exact zero-strength recovery and independent small FP64/native forward and backward checks. Verify tile equality and schedule save/restore before any model fitting.
4. **One matched adaptation pair, only after separate run authorization:** create distinct child identities `u_s_e1000_control_adapt_v1` and `u_s_e1000_egridprior_adapt_v1`, both initialized from the exact same literal parent weights. Keep the full-TRAIN600 case membership, inherited A2-D normalizer, Q1024 query recipe, native objective, work targets, case/query schedule and 100-local-epoch horizon identical. Use FP32, seed0, effective batch48 and microbatch24; each epoch has twelve48-case macros and a final 24-case macro normalized by its actual case count. Reset optimizer and local scheduler state identically using the parent's saved optimizer-group/LR recipe, copy the same parent M/E/context controller state to both arms and continue its scientific clock from 1000. The new lambda schedule uses its own local update cursor and starts at zero; no inherited waiting phase is introduced. Label the comparison a matched adaptation, not an exact resume or continuation of the parent's optimizer trajectory.
5. **Review the pair once:** compare against the unchanged literal parent and the adapted control on the same exposed VAL89 receivers. Retain TRAIN-only diagnostics as training evidence, and label VAL89 as exposed validation rather than independent TEST. No quarter-data selection or new normalization is introduced for this lineage.
6. **Keep later work conditional:** do not automatically extend the horizon, start a portfolio, activate spatial regularization, learn radii, change calibration, add candidate indexing, or run inverse search. Each is a separate decision after the matched-pair review; fixed-layout K reuse can be qualified independently from geometry-design claims.

The 100 local epochs are a proposed bounded adaptation horizon, not a measured optimum or authorization to launch. Report actual case visits, updates, queries and worker time. Do not reinterpret local epochs as original training epochs or infer a new absolute clock from the e1000 checkpoint.

Each arm represents60000 case visits,1300 optimizer updates and 61440000 primary query occurrences, excluding the unchanged auxiliary/response workload. These are protocol arithmetic, not completed work or timing estimates. With no shared fitted prefix, the two-arm total is 120000 visits and 2600 updates. Preserve case/query hashes, equal-case work reductions and once-per-epoch response placement. Retain latest/best states, the declared local 100 milestone and any explicitly requested clean stop; refresh curves every10 epochs rather than create a checkpoint grid.

The parent's scientific sparse age1000 is distinct from the raw controller coordinate, which retains the existing3000 offset; restore and advance the saved coordinate rather than substitute raw1000. Bind sampler epochs 1001-1100 and local adaptation epochs 1-100 explicitly. Both arms use the same case/query streams, while the lambda ramp advances only on completed local optimizer updates.

### Dataset identity and response supervision

This proposal intentionally follows the exact full-TRAIN e1000 U-S parent rather than the default future quarter-data protocol: both children use the parent's exact TRAIN600 membership and inherit the saved A2-D normalizer fitted on that membership. This is a named full-training-data adaptation lineage, not a new quarter-data development result and not a fresh normalization fit. Verify the parent checkpoint and provider manifests before any future run; do not substitute another checkpoint because its case count looks similar.

Use the parent's saved native objective, response addendum, query recipe and work-controller settings in both children. The optimizer and local scheduler are deliberately initialized under one predeclared adaptation config and identically across the pair; the parent optimizer state is not represented as continued or exactly resumed. Save child optimizer, scheduler, sampler, RNG and lambda-cursor state at ordinary declared checkpoints so an interruption resumes the child adaptation faithfully.

Validation is the same exposed canonical VAL89, not independent TEST. M1/M2 detailed evidence remains selected TRAIN evidence because VAL89 contains no such categories. Keep the parent's TRAIN-only A2-D transform fixed in both children; a refitted transform with unchanged scaled weights is not function-preserving.

Use the declared monitoring/latest/best retention and ten-epoch curve refresh appropriate to the authorized adaptation recipe. Do not create extra checkpoint grids, re-evaluate historical models, or start a second membership/normalization arm.

## 21. Evaluation that can reject the idea

### Predictor

Report native u/v/p/omega, fluid/surface/material temperature, q-proxy and independently recomputed module peaks by mean, p90, worst IDs and populated M strata. The E-grid-prior candidate is rejected as a practical recommendation if its field/response losses exceed the predeclared acceptable tradeoff, even when support looks cleaner.

### Organizer

Show low/high-M gate footprints and query-indexed hyperedges, scored E rows, exact-positive selected M/E rows, actually executed M/E value rows, active-role coverage, coefficient mass, weak-positive bins and spatial variation. Report a module-layout response only as a full-model derivative through existing paths; the new fixed-E-grid penalty has zero explicit module-translation derivative at fixed q/domain/radii. Keep diagnostic association separate from physical causal identification.

### Inverse reuse

For fixed-layout heat allocation, compare cached K contraction with fresh native prediction and verify constraints/source IDs. For saved candidate ranking, preserve the original objective and pressure limit, report feasibility confusion and selected-candidate reference regret only when feasible. For geometry sensitivity, compare full derivatives and finite differences without inventing moved-layout truth.

The saved families cannot provide missing geometry-plus-heat endpoints or independent environmental excitation. New physical references, solver validation, broad inverse search and generated designs are separate scopes, not implied by this document.

### Cost

Compare same-function executors separately from the matched trained children. Charge preparation, all-eligible score and E-distance construction, packing, selected and executed wide values, backward/work replay, optimizer and transfers according to clearly stated timing boundaries. Report algorithmic row counters independently from measured wall time. A contended epoch time or lower exact-positive count is not a complete-call cost result.

The retained result should answer whether selection became more structured, whether actual work fell, whether complete cost improved and whether decisions/derivatives became more useful. A success on one axis is not a pass on the others.

## 22. Failure modes and decisions

| Observation | Interpretation | Recommended response |
|---|---|---|
| Gate mass shrinks but positive counts stay high | Attenuation, not useful support reduction | Inspect existing controller mismatch; no sparse-work claim |
| E gates change but the dense backend still executes all value rows | Selected support changed without executed-row reduction | Report selected and executed counts separately; make no speed claim |
| A module-translation derivative is attributed to the new E-grid prior | The explicit prior is independent of module positions at fixed q/domain/radii | Attribute full-model sensitivity to existing paths or revise the feature law in a separately reviewed design |
| Far-field or pressure response worsens | Geometry bias is too restrictive | Weaken/reject the prior; do not impose a local physics claim |
| Peaks worsen while mean fluid T improves | Incomplete field preservation | Report the miss and reject according to declared tradeoff |
| Nearby gates become smooth but outputs lose sensitivity | Overregularization or dead paths | Reduce/reject the stability amendment |
| Same support but unstable coefficient/field derivatives | Masks alone miss conditioning problems | Inspect calibration, normalizers and response compensation |
| Row-max ties dominate derivative errors | Inherited score regularity is limiting | Isolate a smooth-calibration candidate; no blanket C1 claim |
| Footprint candidate list misses active records | Indexed executor does not compute its stated function | Fix completeness or retain full-record reference |
| Sparse values are slower than dense-weighted | Irregular execution dominates | Retain useful model, select honest backend and cost label |
| Heat reuse works but geometry ranking fails | Affine reuse is narrower than design competence | Keep the reusable heat interface; do not claim geometry success |
| Better-looking co-retrieval groups fail physical response | Interpretation metric is not scientific utility | Preserve the negative result; no causal graph claim |

No ordinary negative result authorizes a new model portfolio, unlimited horizon or changes to current formal experiments. A semantic/nonfinite failure stops the affected candidate; a resource stop preserves its exact resumable state.

## 23. Proposed deliverables and storage boundary

1. One explicitly versioned E-grid-prior direct selector with zero-strength recovery of its declared baseline.
2. Fixed adapter frame scales, an externally scheduled strength and a bounded, interpretable access-field visualization.
3. One coherent actual/proxy work controller with truthful per-backend counters.
4. Native numerical/gradient tests and a bounded matched development comparison if later authorized.
5. Fixed-layout K reuse and live geometry-sensitivity interfaces with clear cache invalidation.
6. A compact report separating fields, implicit organization, actual work/cost and inverse-use evidence.
7. An explicit decision to retain the primary amendment, keep an executor-only improvement, or reject the new prior.

No generated payload is needed for this design note. A future implementation must verify the intended /data2-backed roots and shared cache identities, record free capacity, and set retained/temporary/peak budgets and minimum headroom before materializing outputs. Keep selected numerical evidence and checkpoints; remove only round-owned disposable profiling/test outputs after exact dependency and active-use checks. Do not copy datasets or delete historical evidence/migration backups.

A preliminary engineering estimate for this primary E-only amendment, integration and numerical qualification is 3-5 working days for one experienced developer using the maintained runner. This excludes fitting, GPU availability and scientific recovery. The optional candidate-local law would be a separate1-3-week integration effort, not an automatic continuation. Keep the selected 24-case/Q1024/FP32 protocol and report measured complete cost; occupying more VRAM or closing more gates is not a throughput measurement.

## 24. Source map and external precedents

### Repository basis

- [Round11 development report](../../docs/reports/HONF_Round11_Thermal_Development_Report.md): native U-S fields, actual work, decision limitations and executor evidence.
- Local, unpublished Round11 practical-development input: `../HONF_Round11_Thermal_First_Practical_Development_Work_Plan.md`. It informs bounded research and preservation boundaries; this revision does not publish that separate user-owned plan. The tracked report and development protocol retain the durable evidence.
- [Thermal development protocol](../../docs/guides/Thermal_Model_Development_Protocol.md): exact selection/normalization, response exposure and review rules.
- [Continuous direct routing](../../src/honf_forward_core/continuous_joint/routing.py): separate direct M/E gates and local M envelope.
- [Score calibration](../../src/honf_forward_core/joint_native/routing.py): current convex head mixture, bounded scale and eligible row maximum.
- [Query-source reader features](../../src/honf_forward_core/joint_native/readers.py): current learned query-to-source geometric score features, which remain active in this proposal.
- [Thermal native observations](../../Case_ThermalChannel/src/channelthermal/native_observations.py): fixed normalized E-grid coordinates and module-derived native observation features.
- [Thermal local records](../../Case_ThermalChannel/src/channelthermal/local_records.py): adapter feature policy that drops module-distance and occupancy summaries from the unified local E inputs.
- [Continuous primitives](../../src/honf_forward_core/continuous_joint/primitives.py): exact-zero gate, all-eligible guarded normalization and work surrogate.
- [Typed work accounting](../../src/honf_forward_core/continuous_joint/budget.py): actual counts, compulsory floors, surrogate gradients and local schedule.
- [Unified preparation](../../src/honf_forward_core/continuous_joint/unified.py): applied contextual policy, removal of summary paths and prepared-state ownership.
- [Contextual reduction](../../src/honf_forward_core/continuous_joint/preparation.py): coefficient/mass semantics, same-function backends and complete frozen plans.

Source observations describe the inspected design boundary, not current full-data training progress or a replay of private checkpoints.

### Primary external precedents

[Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146/) develops exact-sparse attention/output mappings. It supports distinguishing exact support from dense small weights; it does not validate the proposed geometry bias, Thermal gate, inverse reuse or executor savings.

[E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html) provides a precedent for geometry-aware transformations and permutation/equivariance structure. This channel problem retains its prescribed boundaries and flow direction; it is not assumed to possess arbitrary rotation symmetry.

[Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859) incorporates target derivatives when those derivatives are available. The model-side regularity proposed here is not such physical derivative supervision; finite differences of a surrogate do not create missing physical labels.

All new score amendments, regularizers, reuse policies and candidate-local equations above are project-specific proposals requiring independent tests.

### English visualization index

| Figure | Purpose | Evidence status |
|---|---|---|
| 1. U-S chain with E-grid-prior overlay | Connect applied preparation, direct access and E-conditioned values | Source-derived with proposed overlay |
| 2. Query-indexed hyperedges | Explain overlapping implicit groups without a stored directory | Illustrative schematic |
| 3. Live design reuse | Separate rule reuse from stale coefficient reuse | Proposed workflow |
| 4. Synthetic E-grid bias | Show exact-positive support change without forced nearest-neighbor access | Synthetic arithmetic example |

All visuals are inline English Mermaid source. No new raster, PDF, checkpoint, dataset copy or numerical export is required.

## 25. Document-level verification

For the 23 October revision, Markdown with tables was parsed and rendered to HTML in memory, all 12 local links resolved to this two-document change or existing tracked files, and prose paragraphs were checked for hard wrapping. Both fenced Python sketches parse, and the definition sketch loads in an isolated CPU namespace. Small in-memory FP64 checks verify zero-strength equality and the synthetic closure arithmetic using the revised gate helper; case/update/query arithmetic was checked independently. Original proposal preimages and compact validation receipts are retained locally under /data2, outside Git. These checks do not qualify the percentile/ramp rule, tile execution, saved-cursor inference, literal e1000 adaptation or native model integration.

Earlier in-memory FP64 checks covered the draft's synthetic closure arithmetic, zero-strength identity, an older general geometry-gate sketch, the optional candidate-local reduction and its gradients, footprint-boundary behavior, and smooth log-mean-exp calibration. They do not validate the fixed-radius E-only formula, current cap/ramp schedule, model integration, tile ownership, training behavior, native parity, physical sensitivity or inverse-design success.

The four English Mermaid diagrams are retained as source. Their graphical layout has not been rendered or visually qualified in this documentation task. The round revises the two companion Markdown proposals and retains local provenance/validation receipts; no implementation, native model test, checkpoint, training run, scientific numerical export or GPU operation was performed.
