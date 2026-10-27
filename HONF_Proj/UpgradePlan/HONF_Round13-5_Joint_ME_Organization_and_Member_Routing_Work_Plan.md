# HONF Round13-5 — Joint M/E organizational views and original-member routing

**Planning date:** 26 October 2026. **Repository:** `cosmos2w/ModularDT`, current non-default working branch. **Scope:** extend the working R13-4 reader with E organizational addresses and joint M/E member selection, qualify it on the fixed segmented membership, and compare a bounded pair on TRAIN600. Organizer formation and usefulness take priority, followed by physical-field fidelity; memory and latency remain measured secondary objectives. Preparing this document does not launch training or change the separately authorized R13-4 continuation.

Records form relationships through actual interaction. M and E can both provide organizational views. A query chooses among those views and their permitted original members. Continuous weights learn the relationships; coverage truncation determines the records executed for that query. The organizer is the applied relationship structure and its access rules, not the query softmax by itself and not a pooled feature replacing the members.

## 1. Starting evidence and the specific extension

The [R13-4 report](../docs/reports/HONF_Round13-4_Relation_Guided_Member_Access_Report.md) establishes a useful original-member path: actual contextual relations form the directory, early E candidates are chosen before their member MLP, and one joint physical-field predictor retains scene/control dependence. Paired full/restricted physical training substantially reduces the old approximation penalty. The transferred Guide improves eight of nine VAL89 means against its Full comparison, and the independently initialized Guide has useful initial thermal learning. Preserve these gains, the weighted member-score calibration, prescribed-control ownership and historical checkpoint loading.

The current directory uses M receiver rows only. In M1, the sole proposal equals one, so the early prior cannot change with query location. R13-4 also retains all eligible M. R13-5 removes both restrictions: query-dependent E views can operate even in M1, and one joint member shortlist can contain E only, M only or both. A pure-E final read is a legitimate organized operation; its contextual E states may already contain information propagated from M. It does not mean module controls were ignored upstream.

The first comparison changes neither the two ordered MM→EM→EE→ME contextual cycles nor the general joint physical head. No new output-physics branches, heat-affine corrections, geometry-neighbor mandates, pooled group values, target K, compulsory type winners or forced leaf periods are introduced. The initial address inventory again includes original M/E records for coverage. It is an initial directory, not a claim that retaining every original address is the desired final scale; the last stage explicitly examines shared organization.

Implementation should begin with [current preparation](../src/honf_forward_core/interaction_reused_member_routing/core.py), the [R13-4 reader](../src/honf_forward_core/interaction_reused_member_routing/guided.py), [ordered contextual stages](../src/honf_forward_core/content_hypergraph/staged_attention_v2.py), the [paired native task](../Case_ThermalChannel/src/channelthermal/training/guided_member_routing_task.py), and the [development protocol](../docs/guides/Thermal_Model_Development_Protocol.md). Reinspect live code and user edits before implementation; do not reset or stage unrelated work.

## 2. Build H from the four exchanges that actually ran

Use the captured final-cycle applied gate tensors, retaining their live autograd connections and original receiver/donor IDs:

\[
H^{\mathrm{exchange}}=
\begin{bmatrix}
G_{MM}^{\mathrm{cycle1/mm}} & G_{ME}^{\mathrm{cycle1/me}}\\
G_{EM}^{\mathrm{cycle1/em}} & G_{EE}^{\mathrm{cycle1/ee}}
\end{bmatrix}.
\]

Rows are organizational addresses, ordered as M then E; columns are original members, also M then E. Route names follow the code's receiver-first convention. EM is an actually executed M→E exchange, not a transpose of ME. The two cross blocks therefore have independent provenance and need not be symmetric.

| Block | Receiver/donor state provenance in the final cycle | Shape excluding batch |
| --- | --- | --- |
| MM | Cycle-boundary M receivers and M donors; self excluded | M × M |
| EM | Boundary E receivers and after-MM M donors | E × M |
| EE | After-EM E receivers and donors; self currently permitted | E × E |
| ME | After-MM M receivers and after-EE/FFN E donors | M × E |

This is a stage-labelled reusable relation directory. The blocks were not evaluated at one simultaneous state, and they are not an end-to-end physical influence matrix. Final contextual records supply query keys and original values; the gates retain their actual earlier stage provenance. Applied gates also differ from complete head-specific message transport coefficients, which include attention, donor measures and guarded normalization. Report precisely which quantity H contains.

Handle self access explicitly. The first version preserves current M identity access and the learned EE diagonal without adding another E identity:

\[
H=H^{\mathrm{exchange}}+S,\qquad S=\operatorname{diag}(I_M,0_E).
\]

MM's applied diagonal is zero, so the M identity is an explicitly declared access operation. EE already has a captured diagonal; preserve that coefficient exactly. Do not overwrite it with one or silently add another self edge. Store and visualize the exchange and self ledgers separately. If later evidence motivates additional E residual self access, it is an explicit model amendment with separate accounting, not an invented EE message. An address with no positive eligible retrieval path is excluded from proposal normalization; genuinely empty inventories return zero contexts without a hidden value fallback.

Use `(role, original_id)` as identity. M0 and E0 are different records. Composite identity determines deterministic ties and index mappings; neither padding position nor an untyped integer ID is sufficient. Repeated access to the same original member is coalesced before member scoring, preserving the live sum of every address contribution.

## 3. Joint calibration without type quotas

### 3.1 One shared address scorer

Reuse the existing proposal pair scorer as a single shared function on M and E addresses: normalized query state, normalized contextual member key and the same declared physical geometry features. Its parameters are shared, not copied into two independently competing networks. Initialize compatible weights from the declared parent. This is the smallest implementation change and keeps the first scientific comparison focused; scoring the extra E addresses is real additional MLP work and must be charged. Exact affine factoring/caching is permitted. A smaller shared dot-product scorer is a later measured simplification, not an automatic second architecture or an unreported replacement.

For positive-base-measure eligible addresses, use:

\[
\pi_a(q)=\frac{\omega_a\exp(s_\theta(q,a)/T)}{\sum_b\omega_b\exp(s_\theta(q,b)/T)},\qquad
\omega_a=\begin{cases}1/n_M,&a\in M,\\ v_a/\sum_{b\in E}v_b,&a\in E.\end{cases}
\]

Use stable masked softmax/log-sum-exp and a shared temperature, initially T = 1. Counts and E measures are over eligible addresses of the relevant role; an absent role contributes no terms. The base measure gives equal total role weight under equal scores and valid equal-evidence rows. It is not a quota: learned score differences can move essentially all proposal mass to either role. Preserve the inherited query/key feature normalization. Do not independently center or standardize M/E logits; that would remove learned type offsets. Do not automatically copy the old across-M variance normalization onto the joint scorer, because it can limit the role contrast needed for a legitimate pure-role decision. Observe score ranges and saturation directly instead of restoring a fixed-zero null.

### 3.2 Preserve raw H; normalize a declared retrieval view

The old typewise information measures have totals proportional to M and E counts. They cannot be silently reused as though cross-type competition were already calibrated. Define the first joint member base measure explicitly:

\[
\mu_i=\begin{cases}1/n_M,&i\in M,\\ v_i/\sum_{j\in E}v_j,&i\in E.\end{cases}
\]

M is a discrete object measure; E uses its positive physical cell measure. Physical module areas, cell areas, coordinates and prescribed controls remain unchanged metadata and contextual inputs. These normalized retrieval measures neither replace native physical integration weights nor assign output-physics responsibilities. Exclude zero-measure/ineligible E before normalization, preserve the existing present-M identity convention, and handle absent roles explicitly.

A second calibration issue remains if raw row totals are silently mixed into the proposals. Define r_a = sum_i H_ai mu_i. Raw composition would give an effective address mass proportional to pi_a r_a. For M1/E192, equal-evidence identity-only rows and type-normalized mu yield approximately 99.48% M mass under that raw composition, despite equal total address-role proposals. Under a dense equal-coefficient directory, the same convention gives 50% M mass. Fixed column scaling alone therefore does not resolve both cases.

For the primary model, use one whole-row retrieval normalization:

\[
R_{ai}=\frac{H_{ai}\mu_i}{r_a},\qquad P_i(q)=\sum_a\pi_a(q)R_{ai},\qquad r_a>0.
\]

This is an explicit choice relative to raw `pi H mu`: a proposal weights a normalized organizational view instead of also receiving an implicit row-total multiplier. H remains the authoritative applied-relation evidence; R is the derived live retrieval view. Normalize the entire M+E row once, never the two column blocks independently. Multiply by mu once here, not again after composition. Preserve raw r_a and compare raw versus normalized row contributions on a small TRAIN replay. Do not claim unchanged numerical predictions or exact equivalence to R13-4. Both new comparison arms use this same convention.

Planning-time memory-only arithmetic checks confirmed 50/50 role mass under equal-evidence dense and identity-only fixtures after this whole-row calibration. With strongly E-favoring address scores, the identity fixture also admits a joint95 shortlist containing E and no M. These verify the proposed equations' capability, not trained model behavior. Native TRAIN execution and physical evidence remain necessary.

## 4. One hard selection location: joint95 original members

Initially keep every eligible address in the continuous mixture. Form one normalized P over the union of original M/E members and retain its shortest descending-weight prefix:

\[
C(q)=\operatorname{Prefix}_{0.95}(P(q)),\qquad C_M=C\cap M,\quad C_E=C\cap E.
\]

There is one joint coverage budget. Remove R13-4's compulsory all-M mask and separate E-only prefix. Do not take 95% separately inside each type, reserve one M, enforce a minimum E count or perform another final absolute-weight truncation. Resolve ties by composite original identity. For a valid nonempty P the prefix is nonempty; either type may be absent. A pure-E final read means C_M is empty, not merely that most proposal mass lies on E addresses. Since an E address can reach M through EM, choosing an E address alone does not determine the final member types.

Audit this feasibility on the native rows before prolonged fitting. For any mixture, P(E) = sum_a pi_a R_a(E) <= max_a R_a(E). A pure-E95 list requires P(E) >=0.95, so if every available row has R_a(E) <0.95, no address scorer can produce that outcome on the current directory. Even the necessary mass condition is not sufficient if individual M entries rank above some retained E. Record this bound and the actual prefix outcome for each probe, with the analogous M-only check. This distinguishes a coding defect, proposal-learning issue and a genuine limitation of the composite row family.

Do not weaken useful M→E contextual exchange merely to manufacture pure-E statistics. If persistent cross-block coupling prevents an intended access pattern, investigate one minimal conditional access refinement or source-labelled block-specific view, such as using the existing EE subrelation as its own operation. It must retain actual edges/stage provenance, preserve access to the useful EM view, specify its score/base-measure law and pass controlled/native execution before replacing the initial candidate. No forced role fraction or unorganized value branch is allowed. This is an evidence-driven correction within remaining qualification scope, not permission for a third full-data campaign. The initial full four-block directory remains the first tested formulation.

The remaining members may still be distant from q. Their access is justified by the reused relationships and learned preferences, with physical usefulness measured empirically. No nearest-cell mask, fixed radius or forced geometric partition is permitted. Selecting only E through the EE portion of the directory is legal; it must be observed in actual final member IDs rather than inferred from a row's label.

The candidate mask must precede the member MLP and all query-specific member message work. A restricted call independently computes its candidates, member scores and calibration; it cannot borrow full-pass scores/statistics from paired training. Cached original-record keys/values remain legitimate preparation. Count address scoring, scalar composition and sorting as work, even when some member MLP evaluations are avoided.

### 4.1 Keep the existing typed reader first

Split the admitted IDs by role only after the joint prefix. Retain R13-4's prior-weighted member-score calibration, bounded scores, independent smooth read strengths and single joint physical head:

\[
p_i^t\propto P_i\exp(\ell_i^t),\quad i\in C_t;\qquad c_t=\rho_t\sum_{i\in C_t}p_i^tV_t(z_i),\qquad \widehat u(q)=F(q,c_M,c_E).
\]

Within each nonempty C_t, p is conditionally normalized. Keep the inherited member temperature at one in this comparison, as assumed by the formula; a nonunit-temperature variant must explicitly temper both score and log prior. Empty C_t produces a genuine zero context before logarithms, entropy or reductions. A pure-E read therefore invokes no M member scorer/value reduction for that query. M may still have participated in contextual formation. All fields use the same joint predictor.

Be precise about competition: the first implementation makes M/E compete for candidate admission. Surviving types keep independent conditional preferences and rho amplitudes. Consequently, a small surviving type mass can regain a substantial conditional contribution. Log the retained joint masses sum_{i in C_t}P_i alongside p, rho, contexts and fields. Do not describe this as joint final amplitude normalization. If that reinflation causes a measured problem, the simplest later alternative is a joint posterior proportional to P_i exp(ell_i), or an equivalent posterior role-mass multiplier before typed reductions. It requires no new controller, but changes the read law and is not silently added to the first comparison.

### 4.2 Continuous training and recoverable task feedback

Retain the paired physical objective:

\[
\mathcal L=\tfrac12\mathcal L_{\mathrm{physical,all\ eligible}}+\tfrac12\mathcal L_{\mathrm{physical,joint95}}.
\]

Both executions share parameters, original scene preparation and training queries. The full execution uses all relation-eligible members and no hard address/member truncation. Its native physical objective supplies feedback through positive members omitted by joint95; the restricted objective teaches prediction under actual candidate admission. Charge both decoder executions and any checkpoint recomputation. Full training reference work is absent from deployed restricted predictions and is not an unorganized value bypass.

Do not differentiate through the integer prefix by assertion. An exactly zero applied interaction gate does not acquire a direct recovery derivative merely because a full reader exists. Track actual contextual and decoder task paths, not only analytic derivatives or positive-floor weights. No entropy penalty, forced support fraction, target pure-E frequency, fixed-null competition, delayed E permission or new auxiliary controller is included initially. If a concrete learning defect remains, correct the identified equation/calibration/indexing or add one justified task-derived refinement within the existing qualification budget; do not respond by opening an unlimited arm sweep.

## 5. Geometry and self access must remain interpretable

Existing address inputs already contain positions, relative geometry and contextual records. Start with those. If TRAIN-only diagnostics show a specific need for an additional spatial preference, allow one simple soft distance bias with explicit units/scale and a documented coefficient. It may guide access but must preserve distant eligibility; no hard neighborhood or minimum local fraction. A retrieval-only bias is labelled as query-access bias. If a bias is instead introduced into contextual attention, it must actually run there and its changed applied relations must be recaptured into H. Never color a post-hoc geometric graph and describe it as learned exchange.

Inspect self-only versus off-diagonal contributions. With continuous positive address weights and identity/self paths, full support can cover the whole original inventory. The organizer's useful work must therefore appear in relational weights, changed candidate IDs and task effects. E identity matching can be useful but is insufficient evidence that exchanged EE/EM relations organize access. Compare self-only, actual exchange and legal row reassignment on a bounded TRAIN panel, with actual context held fixed for decoder-only interventions. Removing an applied exchange message is a distinct intervention requiring re-preparation of all dependent states and consistent reuse of the changed relation.

## 6. Minimal primary experiment and qualification

| Arm | Addresses permitted | Directory and joint calibration | Candidate/execution policy | Training |
| --- | --- | --- | --- | --- |
| **M-joint95** | M only | Actual M receiver rows; same whole-row calibration and shared scorer | One joint M/E 95% member prefix; no compulsory M read | Paired full/joint95 physical losses |
| **ME-joint95** | M and E | Four actual blocks plus explicit self ledger; same calibration/scorer | Identical one joint95 member prefix | Identical paired objective |

This pair isolates the extra organizational view family under the new admission rule. Existing R13-4 Guide99/keep-all-M evidence remains a historical reference; do not automatically train a third old-policy baseline. The arms share trainable parameter definitions and initialization; their address eligibility and resulting executed work differ. Report those differences rather than claiming equal compute. Optional geometry or read-law changes must be identical in the pair or explicitly replace the original experiment after diagnosis, never be introduced only in the favored arm without attribution.

**Before fitting:** execute focused CPU arithmetic/core tests and native prepare/decode/backward on a few fixed TRAIN cases. Check four-block shapes and directions, EE diagonal policy, composite IDs, dense/identity calibration, one joint coverage budget, zero-width/absent-role handling, member permutation/ties, and actual pre-MLP selected execution. Include controlled pure-E and pure-M fixtures, plus an E-only core scene without M; synthetic capability tests are labelled separately from physical accuracy. Check that M1 can form query-dependent early candidates when E address relations/scores differ. A failing equation or wired-out E path is corrected before any long fit.

**Segmented qualification:** reuse the fixed input-only TRAIN150/DEV22 membership and its TRAIN-only normalization. Use fresh qualification initialization shared between compared arms; never silently cross full-data normalization or promote segmented weights. Begin with six epochs per arm to pass the previously observed e2/e3 failure onset. If needed, resume the selected lineage through twelve for persistence and actual checkpoint recovery. Review each initial epoch using inexpensive inline health callbacks. These endpoints are observation windows, not forced warm-up or permission schedules. The entire round, including retries and conditional changes, has at most 500 segmented epochs; it should normally use a small fraction. A healthy but broad early candidate or a not-yet-frequent E view does not justify either exhausting that ceiling or declaring architectural failure.

**Full TRAIN600 pair:** after learning-path and recovery review, use the retained R13-4 independent `epoch_0100_model.pt` as the common fixed transfer source, read in place through `thermal_r13_4_guide_train600_100_v3/checkpoints`. It is separate from the active run's changing latest/best aliases. Initialize compatible encoder, keys, values, proposal, member reader and joint head identically, and retain compatible named optimizer, RNG, sampling and TRAIN600 normalization state. List every intentional numerical/state difference, including the new address law and joint measures. These are named transferred R13-5 development lineages, not an unchanged v3 resume. If the architecture requires a narrow state mapping, test it and document the mapping; do not reset unrelated state to make loading pass.

Fit at most 100 additional full-data epochs per explicitly selected arm, 200 aggregate for the primary pair. Label the parent age and new reader's additional training separately. Match case order, query draws, native physical roles/weights, effective batch, dtype and schedule. A scratch pair is a possible replacement only if a real transfer incompatibility is diagnosed and the replacement is documented before launch; do not add a scratch campaign or third delivered model automatically. The active R13-4 continuation remains untouched. A reserved or available GPU is not permission to exceed epochs; longer training requires user instruction. A first100 endpoint is preliminary development evidence, not convergence or an architectural verdict.

## 7. Diagnose early, repair, and keep exploring within scope

Use existing health callbacks and ordinary clean-stop/recovery mechanisms. In the first six to twelve qualification/early full-data epochs, record compact fixed low/high-M TRAIN observations each epoch; thereafter use the existing ten-epoch health/review cadence and final endpoint. Save/restore diagnostic RNG, mode and provider state. Capture only a bounded query trace while computing the declared native objective; do not request every native receiver's full graph tensor for a startup probe.

Observe proposal score ranges and address-role totals, effective addresses, raw row masses, exchange versus self contribution, four-block gradient provenance, retained joint role masses, candidate IDs/counts, full-reader preference captured, typed strengths and conditional preferences, original-value task gradients, fieldwise losses and valid prescribed-input sensitivity at fixed q/geometry. Include M1/M2/M5/M12 TRAIN health cases and the already fixed physical-detail panel. Query perturbations and control perturbations answer different questions and are labelled separately. Do not impose heat-independent flow, Thermal-specific analytical responses or compulsory per-output role assignments.

Pure-E, pure-M and mixed candidate lists are all valid. Zero M-value gradients for a pure-E query, an unused block at one query, and absent-role gradients are expected possibilities. Health checks must inspect native scene/control dependence through the paths that are actually used, including M information propagated into E. Positive address weights alone are not evidence of useful gradients. Conversely, an apparent all-E winner caused solely by the 192-to-M count imbalance or index collision is an implementation/calibration defect, not learned specialization.

Nonfinite values, broken provenance/ID mappings, a nominal E branch disconnected from native task feedback, or widespread loss of previously used input dependence triggers an immediate clean stop of the affected owned run with its failing numerical state retained. Diagnose the state before more fitting even when total loss falls. Inspect eligibility, joint versus typewise normalization, temperature/logit saturation, self-row dominance, indexing, checkpoint mode capture and gradient connection. Correct one concrete mechanism and requalify within the remaining allowance. Do not keep replaying an unexplained failed launch, silently accept a coordinate-only predictor, or turn the first disappointing metric into the final delivery.

Broad active routing, a legitimate role preference or mixed early field errors call for scientific analysis. Inspect the mechanism and choose a bounded corrective test; do not invent numerical admission bands or force a favorable topology. If the path is healthy, full-data learning is the appropriate next measurement. At the authorized endpoint, provide concrete trajectory evidence and the most useful continuation/alternative; do not claim a mature failure merely because100 new epochs did not form compact groups. If scope or resource limits prevent a required correction, state the unresolved mechanism and preserved resume state precisely.

After each fitting launch delegate read-only monitoring to an explicitly selected Luna agent with a minimal run-specific handoff: exact owned run, epoch stop, logs/checkpoint locations, existing health/clean-stop signals and notification conditions. One startup inspection establishes real epoch timing and an ETA. Inspect near 80–90% of expected remaining duration or a declared necessary milestone, with earlier checks only for concrete failure/delay. Inline health callbacks catch rapid collapse without repeated parent polling. Luna stays quiet until completion, failure, meaningful delay or user action; it cannot launch, restart or extend runs. The main agent performs necessary independent work or waits. No new daemon, reservation or admission framework is added.

## 8. After member selection works: address truncation, then shared organization

### 8.1 A bounded address95 diagnostic

First retain the continuous address mixture as the reference. Once the member-selection candidate has real learning/fidelity evidence, evaluate address truncation on at most four fixed TRAIN cases and the fixed detailed panel at the retained endpoint; use no new fitting initially. Rank addresses by their actual transported member mass. Under the primary row-normalized R, this equals pi; under any raw-H alternative it is proportional to pi_a r_a, not pi alone.

Measure four executions: no cuts, member95 only, address95 only and address95 plus member95. Recompute the restricted directory mixture and candidates independently. Track omitted address path mass, composed member-prior mass relative to the original uncut mixture, full-reader preference, per-type candidates and native field differences. Two compatible95% stages can retain only90.25% of path mass; under other definitions even that simple product is not the appropriate accounting. Never advertise two independent cuts as one5% omission or as a physical-error bound.

Prefer a single combined5% omission budget if early address restriction is retained for a later trained model. One simple compatible allocation is address retention alpha and conditional member retention beta satisfying alpha*beta >=0.95. If the requested address95 experiment remains at alpha=0.95, maintaining95% total path retention leaves no further independent member omission allowance in that conservative product calculation. Alternatively measure the exact combined member deficit against the uncut mixture. Do not silently keep two5% budgets. Address selection still requires scoring the full initial address bank; it can save directory expansion, not those already computed address scores.

If address95 damages useful fields or role access, preserve the comparison and keep the continuous-address candidate as the result while diagnosing which rows carry the lost information. Do not discard the working member-selection model. New address-truncated fitting is a later explicitly selected experiment requiring explicit user authorization and its own stated epoch allocation; it is not an automatic third full-data run in this plan.

### 8.2 Organize overlap without pooling values

At the retained endpoint, measure overlap separately within and across M/E address roles using typed original IDs, support Jaccard, normalized-weight similarity and shared versus address-specific tails. Equal-looking spatial maps are not enough to declare two organizational views redundant. Include original self contribution, block/stage provenance and query usage in the interpretation.

Start with exact operations: deduplicate member visits, share repeated posting/index layouts while keeping each row's live coefficients, and aggregate proposal mass for genuinely identical retrieval rows in a no-fit composition check. If distinct address keys still require separate scores, charge them. Exact index sharing does not imply fewer original records or fewer learned organizational views. Preserve gradient attribution to every live contributing row, including rows that may diverge during later training; do not detach/freeze them into a shared constant.

For approximate consolidation, first produce one bounded relationship-space reconstruction/access diagnostic and native field comparison with the original directory. Retain original members and per-row provenance; do not invent new positive access across a zero actual relation or replace members with pooled group values. A learned consolidation mechanism that changes contextual exchanges must run in those exchanges and be recaptured before reuse. This plan requests the redundancy measurements and feasible exact reuse, plus a concrete proposal if meaningful approximate sharing exists; it does not authorize a new merge hierarchy, controller or unbounded retraining campaign. Keeping all original addresses is not declared optimal merely because the first endpoint is reached.

## 9. Evidence and readable visual delivery

The first result axis is useful organization: independent E views, query-dependent M1 access, legitimate pure-role/member combinations, reused off-diagonal relationships and changes in actual eligible/executed IDs. Report address type/role mass separately from final member type and rho. Show captured reference preference and field effects alongside candidate counts; never use sparse support alone as the quality metric. A zero observed pure-E frequency is a diagnostic question about cross-block relations, calibration and task needs, not permission to force a quota or claim that the code lacks the capability when controlled execution demonstrates it.

The second axis is physical-field reconstruction: complete native VAL89 mean/p90/worst errors, per-M strata and original-module surface/material/peak detail. Reuse canonical membership excluding duplicate0273 and train-only normalization. Keep all fields visible under one joint predictor. Existing historical Run1404/U-H/Relative/R13-4 artifacts are context references, not matched new architecture controls. Use the selected fixed TRAIN panel for intervention fitting/diagnosis, never choose cases from validation error or add difficult VAL cases to TRAIN. Saved valid TRAIN response references may be evaluated on a small panel; no new solver or inverse campaign is included.

Use Python to produce a compact English figure set from retained arrays, preserving the user's preferred parallel spatial-map style:

- **Same scene, different organization:** R13-4 M views, R13-5 M views and R13-5 E views on identical native coordinates, with selected actual relation rows, module circles/IDs, address stars, common q and explicit stage labels. Existing1404/U-H panels may be linked/reused as historical context without another baseline replay.
- **M1 is now query dependent:** three fixed queries showing joint address masses, relation-derived prior, early candidate M/E IDs and final typed preference/strength. Mark an actual E-only example when present; otherwise show its observed role composition and the separate controlled capability test honestly.
- **Local and distant cooperation:** one input-fixed multi-module scene with M/E entry views and final reads, strong/shared/weak tails, plus an applied-message removal/re-preparation or decoder-only reassignment with clearly distinguished semantics. Avoid an unreadable all-edge graph.
- **Physical quality and training:** matched full histories, native field/residual maps and original-module detail on TRAIN0003 and VAL0274/0293/0678, with a compact per-field/tail table. Pair field errors with actual address/member work. Keep address-cut and overlap results on one small optional page rather than a figure portfolio.

PDF masters have small inline PNG companions and a short figure index. Inspect the selected figures, decode PNGs, resolve Markdown links and retain quantitative captions immediately below them. Numerical provenance identifies source checkpoint, transfer age, partition, original IDs, units, controls and valid physical references. The English report begins with actual positive progress and then concrete remaining mechanisms for organizer, predictor and efficiency; no repeated vague non-proof language. No new inverse result is implied.

## 10. Implementation, resources and completion boundaries

Add one cohesive, explicitly versioned core/adapter capability to the maintained runner and evaluation path. Preserve v1/v2/v3 equations and historical loaders. Extend prepared organization with live H column blocks, composite address/member inventories and stage/self metadata; reuse existing batched products and query tiles. Do not allocate a query×address×member tensor just to expand support or count work. At E192, dense directory storage itself is small, but Q(M+E)^2 composition and the extra address MLP are real costs. Charge preparation, all four captured blocks, proposal scoring, indexing/sorting, selected member scoring, projections/reductions, paired backward/recomputation, full runtime and allocated/reserved memory separately.

Focused tests cover the new behavior: four-block direction/provenance, no EE self double count, joint base measures and row calibration, one joint95 selection, original-ID permutation/ties, pure-E/pure-M/missing-role execution, M1 query-dependent capability, live native task gradients through actual paths, mode-safe checkpoint recomputation and source/continuation state mapping. Execute relevant existing tests and real native qualification. Tests/preflight assertions supplement execution; they do not mock or replace it. Reuse exact shared caches and avoid unrelated broad refactoring.

Before output materialization verify `/data2` is mounted, the `diagnostics` and `Trained_Results` symlinks resolve to their declared `/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj` roots, and any shared cache recipe/geometry identity matches. Record live free capacity, a hard budget of at most4 GiB new retained payload and512 MiB disposable temporary demand, and minimum free headroom of100 GiB on DATA2 and20 GiB on the home filesystem. All generated arrays, figures, runs, snapshots and temporary caches stay on DATA2; no dataset, parent checkpoint or catalogue copies. Reduce materialization if the measured peak would exceed the budget. Keep latest, one best-field alias, declared100-epoch milestones and requested final/clean-stop states; refresh curves every10 epochs and at completion, preserving full histories, validation cadence and tqdm output.

Training launches must preserve active R13-4 and unrelated jobs. Inspect actual device/process ownership before launch; no automatic substitution for a user-specified device. Follow the host's existing blocked-device policy rather than adding another reservation system. Monitoring does not authorize a restart or epoch extension. This is a new bounded round; R13-4's unused allocation is not silently inherited.

Default-deny new hashes, baseline snapshots/freezes, device locks, admission services and defensive blocking gates. A proposed exception must name a concrete failure and demonstrate why ordinary Git history, existing IDs/manifests, types, atomic writes and focused tests are structurally insufficient. Preserve current security, data/TEST separation, checkpoint, ownership and upload safeguards. New blocking gates belong only at irreversible changes, cross-system mutations, security boundaries or formal releases. Health observations and research bands are measurements reviewed for diagnosis, not new admission infrastructure.

Closeout includes verified cleanup of round-owned disposable caches and superseded exports while retaining scientific checkpoints, numerical evidence, source helpers, required provenance and unrelated/active artifacts. Report exact files/bytes removed, retained payloads and remaining capacity under the repository cleanup rules. Keep generated evidence and one-time tools in ignored DATA2 paths. Commit durable source/tests/configs/docs only to the current non-default branch, inspect/fetch the remote, audit the entire outgoing commit range including deleted earlier additions, run the existing pre-push safeguards and verify local/remote tips match. Stage no unrelated work.

Deliver the coherent implementation, measured qualification/recovery, the two bounded full-data trajectories, organizer/field report and figures, small later-stage address/overlap diagnostics, and tested explicit launch/resume commands. If an anomaly interrupts a measurement, retain the failing state, diagnose and attempt the narrow correction within scope before delivery. The research outcome is a traceable account of how M/E views formed by real interaction govern original-member access and physical prediction, including the next concrete improvement where the measured organization remains broad or inaccurate.
