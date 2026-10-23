# HONF Round8 — learn budgeted native hypergraphs, then execute their member lists

**Status:** proposed development and evaluation plan; none of the proposed learning or performance results has been measured. **Reviewed repository:** `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `782ea56ceba60da0582203bc04acc93bb53889ab`. **Primary evidence:** the complete `HONF_Native_Token_Retrieval_Round7_Report.md`, the original Round7 plan, and the source inventory in §2. **Review boundary:** source and report review, not an independent execution of the private checkpoints or inspection of locally ignored figure masters.

This plan changes the prospective research question and admission policy. Preserve Round7's negative decision, receipts, code, checkpoints and numerical tolerances unchanged. Do not fabricate a positive Round7 admission receipt, relax its validator, or use a new experiment name to overwrite its evidence. A separately versioned Round8 family and research authorization are required.

## 0. Decision in plain language

Retain A2-D's contextualizer, original native token records, query-dependent nonlinear values and dataset-owned field laws. Replace Round7's unbudgeted union of independently permissive heads and seeds with a **learned, shared execution support**. The graph still forms during scene preparation from contextual attention; queries dispatch to native-seeded member lists and evaluate the original records individually. Train accuracy and actual access cost together. Organize execution around repeated member lists rather than an arbitrary flat list of individual pairs.

The first Round8 hypothesis is deliberately specific: **one native-seeded hyperedge is dispatched per query; that hyperedge can contain many original module tokens, with learned variable membership, plus the unchanged local protection.** Different hyperedges overlap. Their number follows the actual input module count, not six latent slots or a fixed spatial anchor catalogue. One dispatched list is not one source, one averaged vector, or a rank-one value representation. This restriction bounds the number of execution families and removes the combinatorial union that defeated Round7. Whether it is sufficiently expressive must be learned and tested; do not call it a general solution in advance.

The core experiment is four continuations: matched D-control and budgeted native-hypergraph G for each dataset. Give both pairs real learning. Preliminary speed, near-full birth support, or failure of an optional optimized backend must not by themselves cancel the learning experiment. A correct dense hard-forward reference is an acceptable research-training backend while the grouped executor is qualified. It is not sparse training and must be reported accordingly.

The deliverable is a measured answer about learned native retrieval, not another collection of startup qualifications. A negative trained result is acceptable; a zero-scientific-update closeout is acceptable only for a specific unresolved semantic/safety failure, unavailable required source, resource exhaustion, or an explicit user stop.

## 1. Diagnosis of Round7: distinguish implementation, method and gate

### 1.1 What the evidence supports

The report records ordinary all-open bitwise native recovery on eight TRAIN fixtures and matching native AdamW boundaries. Original token IDs reach actual nonlinear value rows. These observations and the inspected source support the conclusion that the principal retrieval construction was implemented substantially as specified. They do not establish that every backend is numerically qualified.

The strongest permitted query threshold retained 97.4436% of Thermal M12 pairs and 99.9048% of Wind M30 pairs. At default thresholds, complete high-M Q8192 G/D0 ratios were 1.171/1.132 for Thermal B1/B8 and 1.184/1.139 for Wind B1/B8. Scientific continuation updates were zero. These are untrained-policy execution results at D0 e500, not a failed trained sparse predictor.

A short pre-fit attribution check may additionally hold the Round7 IDs fixed and replace positive fractional permissions by one, alongside the original fractional and all-open functions. It isolates survivor attenuation from omission at fixed weights, not trained retrieval quality. Use at most one native TRAIN scene per dataset and preserve ordinary FP32 backend discrepancies separately.

The plan itself fixed member thresholds at .20, allowed only query thresholds .10/.20/.35, and permitted a stop when none exposed an execution opportunity. Codex followed that provision. The gate was appropriate for deciding whether to deploy the *unadapted rule*, but too restrictive for answering whether a different, explicitly budget-trained graph can learn useful support. Round8 must separate those questions in executable code.

### 1.2 The important methodological defects

**Union inflation.** `planning.py` takes a maximum over preparation heads, expands all query-selected seeds, then keeps a donor if any decoder head has positive permission. Many moderately selective paths can produce almost full final support. Sparsity of a single attention map is not sparsity of the unique nonlinear value rows.

**No incentive targeting the deployed work.** The planned task-only continuation adds no work objective, expected-support constraint, or mechanism tying density to a compute budget. Even had it run, reducing an error could reopen paths. A connected task gradient does not imply a pressure toward economical support.

**Poor learning signals through the original rule.** Quintic permissions are flat at both endpoints. Max-product unions route gradients through winning paths. Every row maximum has relative density one. A nonzero gradient in one isolated test does not demonstrate useful gradients for removing all the paths that jointly keep a donor alive, or for reopening an omitted donor.

**Attenuation without omission.** Round7 multiplies signed Thermal coefficients by fractional permissions. A positive permission still executes the expensive value. This permits physical predictions to change substantially while work stays almost unchanged. In the untrained Thermal0631 diagnostic, fluid-T RMSE changed from 2.06823 to 3.59669 despite nearly full support. Do not infer that this scalar change is entirely caused by one implementation detail; it motivates separately measuring support removal and survivor attenuation.

**Execution overhead remains real.** Tiled max-product expansion still forms `[B,H,Q_tile,seed,donor]` intermediates. The selected reader uses query-sized `nonzero` compaction, packed values and `index_add`. `build_native_retrieval_hypergraph` also reconstructs inspection probabilities unconditionally. The current code is not simply a mask around the original dense reader, but it still pays substantial extra work and memory.

**Numerical scope is unresolved, not a universal veto.** Forced gathered all-open misses strict FP32 comparisons in seven of eight native fixtures; ordinary original-reader recovery passes. Same-permission selected/dense agreement uses a related gathered reduction and is not independent original-reader parity. Preserve these facts. Determine whether any new error is a mathematical mismatch or a floating-point execution difference before deciding what can be trained or deployed.

### 1.3 What not to conclude

Do not conclude that native-token retrieval cannot be learned, that the contextualizer must be replaced, that only pooled representations can be efficient, or that a more lenient speed threshold alone solves the problem. Round7 did not train the proposed policy. Conversely, do not promise that learning will make the existing union fast: even an injected 50% support remained slower in several B1/thermal cells. Representation, learning and execution all need explicit evidence.

Wind source identity is now resolved by Round7: the actual provider uses stored native cell-centred OpenFOAM velocities from `family_volume/U.npy`. Preserve this correction. Thermal remains its separate packed-native benchmark and saved low-fidelity response records. Do not repeat the earlier mistaken Wind “analytic-wake/shared-grid” description.

## 2. Source review and reuse boundaries

Paths are relative to `HONF_Proj/` unless stated otherwise. Function names identify the intended boundaries; verify any later user changes before editing.

| Source inspected at 782ea56 | Relevant observation | Round8 action |
|---|---|---|
| `src/honf_forward_core/native_retrieval/hypergraph.py` | Row-relative density, endpoint-flat permissions, explicit native seed IDs, unconditional inspection probabilities | Preserve v1. Add learned binary memberships with a shared execution view; make optional probability/trace exports lazy in the new path |
| `src/honf_forward_core/native_retrieval/planning.py` | Head maximum, tiled seed–donor product, final head union, `torch.nonzero` | Do not inherit this union as the new planner. Dispatch one native group and retrieve its native member indices; retain v1 for replay |
| `src/honf_forward_core/native_retrieval/reader.py` | Original nonlinear values are retained; sparse scalar coefficients multiply fractional permission; information weights use a zero-excluding log-permission path | Reuse the value formula, not the fractional execution semantics. Add an independent dense binary-mask reference and a zero-sensitive training surrogate |
| `src/honf_forward_core/native_retrieval/core.py` | All original parameters retained; only age buffer added; no direct E consumer | Reuse parent migration and adapter integration. Add small route-calibration parameters and a versioned budget state; do not silently enable QE |
| `src/honf_forward_core/native_retrieval/route_capture.py` | Actual MM/ME Q/K/geometry recompute raw scores with stage and identity binding | Reuse opt-in capture; request only what the active consumer needs. Keep preparation-score stage distinct from final-value stage |
| `src/honf_forward_core/native_retrieval/contracts.py` | Native IDs, stores, query plans and stale-state ownership | Reuse concepts and tests; add seed-dispatch/ragged-list contracts without recursive guard work inside tensor tiles |
| `src/honf_forward_core/content_hypergraph/readers_v2.py` | A2-D's original query–module nonlinearity and signed Thermal head | Preserve parameters, geometry features and head laws; no cheap completion, no pre-query value pooling |
| `src/honf_forward_core/content_hypergraph/decoding.py` | Local pack stores an input-derived compact window before the separate measure multiplier | Preserve local branch. Identify direct-value protected IDs separately from local-branch weighted contribution |
| `src/honf_runtime/native_retrieval_protocol.py` | Exact e500 source contract, fixed thresholds, original case/query/optimizer state | Reuse source identity; new Round8 schema owns budgets and learning schedules, not a modified Round7 schema |
| `tools/native_retrieval_train.py` | Positive Round7 cost receipt is required even to start scientific fitting | Add a new research-admission entrypoint. Retain old CLI behavior and negative receipt rejection |
| `docs/guides/Native_Token_Retrieval.md` | Correctly separates M-value consumption from E metadata | Preserve this distinction in every new guide and report |

The required architecture remains a single jointly trainable A2-D-derived field model per dataset. Do not reintroduce a frozen D-sep, Stage-A, second full predictor, H6 bank, moments, learned value prototype, inverse generator, utility MLP or rank-r completion.

## 3. Mathematical model: adaptive native hyperedge formation

### 3.1 Native token store and graph meaning

Let the prepared scene contain individually retained module records and environmental records:

\[
\mathcal M=\{(\mathrm{id}_i,x_i,z_i,\mu_i)\}_{i=1}^{M},\qquad
\mathcal E=\{(\mathrm{id}_j,e_j,v_j,\nu_j)\}_{j=1}^{E}.
\]

These symbols abbreviate the full existing record: presence, raw features, characteristic lengths, frame and native measures remain present. Contextualized `z_i` or `v_j` can contain information from other records; retaining identity does not make a latent state a pure local physical quantity.

Use the existing two-cycle MM → EM → EE → ME contextualizer and native encoders unchanged. Define one seed family for each actual present module. A group stores a seed ID, typed member IDs, route-score provenance, membership logits/probabilities, hard memberships and execution metadata. It stores no averaged value for the decoder.

The primary field consumer remains M-only, exactly as A2-D. E continues to influence contextualized M through existing exchanges. E-side associations can be exposed through an explicit diagnostic request, but there is no E sparsity reward, direct-E savings claim or “trained QE” claim without an actual QE consumer. Do not spend the main round adding that separate architecture. This is a stated scope, not an excuse to rename E metadata as consumption.

### 3.2 Shared physical memberships, not independent-head OR

Let \(L^{MM}_{rsi}\) be the captured final-cycle raw score before log measure, where r indexes preparation heads. Create a single execution-membership logit per seed and donor. Preserve all original headwise maps for optional inspection, but do not union their positive supports.

A small initial parameterization is:

\[
\bar L_{si}=\sum_r\omega_r\bigl(L^{MM}_{rsi}-\max_{j\in\mathcal V_s}L^{MM}_{rsj}\bigr),
\qquad \omega=\operatorname{softmax}(a),
\]
\[
\ell_{si}=\kappa_M\bar L_{si}+b_M,\qquad
p_{si}=\sigma(\ell_{si}).
\]

Here \(\kappa_M>0\), b_M and the head-mixing logits a are shared scalar calibration parameters, not a new input-dependent importance network. Initialize the head mixture uniformly. Use a bounded positive scale in [0.25, 4], initialized at 1, for both new calibration scales; initialize b_M on input-only TRAIN score statistics to approximately 0.98 expected valid nonself membership. Calibrate once before the scientific branch; no target, current heat, DEV error or inverse objective enters this calibration. The exact value is a birth condition, not a final sparsity target.

Invalid donors have p=0. For module retrieval set p_ss=1 and hard b_ss=1; contextual MM self-exclusion stays unchanged. Empty MM rows are handled without a maximum over an empty set. Use actual presence/measure eligibility consistent with the original physical and information paths; do not silently conflate zero information measure with an absent physical heater. Add a synthetic check for that distinction even if ordinary native data have positive module measures.

The same generic mechanism may describe ME memberships in a future value-consuming mode, with separate type calibration. In this M-only experiment ME is not charged as sparsified work and unused ME gate parameters must not be instantiated.

### 3.3 Binary forward memberships

For training, use binary member choices with an explicit differentiable estimator. One declared logistic-noise construction is:

\[
\epsilon_{si}=\log u_{si}-\log(1-u_{si}),\quad u_{si}\sim U(0,1),
\]
\[
B^{hard}_{si}=\mathbb 1[\ell_{si}+\epsilon_{si}\ge0],\qquad
\widetilde B_{si}=\sigma((\ell_{si}+\epsilon_{si})/T_B).
\]

Sample one membership matrix per scene preparation, not independently for each query. The group can then be reused for all queries dispatched to it. At deterministic evaluation use \(B_{si}=\mathbb1[\ell_{si}\ge0]\). Self and invalid entries override the stochastic construction by their exact native rules.

Use straight-through tensors only during designated router-learning steps:

\[
B^{ST}=B^{hard}+(\widetilde B-\operatorname{stopgrad}\widetilde B).
\]

The parenthesized subtraction is important for exact forward zeros. The forward field sees binary permission, not fractional attenuation. This estimator is biased and is not the mathematical derivative of a discrete threshold. State that clearly in code, tests and reports.

Stochastic probabilities and deterministic final support are not interchangeable. Measure both throughout learning; a good expected budget cannot qualify a near-dense deterministic endpoint.

### 3.4 Query dispatch reuses A2-D scores

Let \(l_{qhs}\) be the original A2-D query–module raw scores before log measure. Form a scalar native-seed dispatch score using a separate convex mixture of decoder heads:

\[
 d_{qs}=\kappa_Q\sum_h\xi_h(l_{qhs}-\max_j l_{qhj}),
 \qquad \xi=\operatorname{softmax}(c),\quad \kappa_Q>0.
\]

This is a small calibration of existing scores. Keep the original four-head query–token information weights unchanged inside the final selected reader. Sharing *execution support* does not delete the four value/attention heads or claim that different routes' head labels have matching meanings.

Training dispatch is categorical over actual valid seeds, \(S(q)\sim\mathrm{Categorical}(\pi_q)\), \(\pi_q=\operatorname{softmax}(d_q)\). A Gumbel-softmax straight-through estimator supplies router gradients. Deterministic evaluation selects the maximum-scoring seed. Exact ties use an input-geometry canonical order and are reported as a piecewise boundary; native IDs are labels for correspondence/RNG only, never learned scalar inputs. For geometrically indistinguishable exact ties, use an explicitly defined tied-union reference or report the indeterminate case rather than inventing an ID-based physical preference. Do not claim global differentiability.

A query is dispatched to one input-native group in the primary experiment. It can still read up to all M original values if its selected group requires them. The expressivity restriction is a shared support pattern, not a shared field value. No tensor averaging of query coordinates or token contents occurs.

### 3.5 Final support and local protection

Let \(P_{qi}\in\{0,1\}\) mark IDs whose original compact local window is positive. It is an input-defined direct-value protection set. Preserve the existing local value branch and its smooth weights separately.

\[
 g^{hard}_{qi}=P_{qi}+(1-P_{qi})B^{hard}_{S(q),i},
 \qquad \mathcal S(q)=\{i:g^{hard}_{qi}=1\}.
\]

Thus values from a selected hyperedge and protected direct exceptions are deduplicated. An exception is evaluated by the **same original nonlocal value function**, not a cheap base plus expensive correction. The separate original local branch remains an unchanged branch with its own counted rows.

All decoder heads share this physical support. A source is evaluated once per query. Group membership may overlap across seeds, but no query takes an unrestricted OR across all seeds and heads.

The protection boundary is discrete even though the old local contribution is smooth. Report possible support-switch seams and do not claim continuous inverse-design geometry derivatives. If the direct-protection overhead is high, report that lower bound; do not remove protection mid-run or expand the local radius to manufacture an accuracy result.

### 3.6 Keep the original nonlinear native values

On the selected records evaluate the exact original formula and parameter/bias placement:

\[
 m_{qi}=W_o\operatorname{SiLU}(W_q q_{state}+W_m z_i+W_\gamma\gamma(q-x_i))+b_o.
\]

Only final receiver-dependent reduction combines values. No group centroid, average, moment, prototype, cached edge embedding or thermal low-rank mode substitutes for a native value. Reusable tokenwise projections are allowed, since they preserve individual identities.

For ordinary information weighting:

\[
 \alpha_{qhi}=\frac{g^{hard}_{qi}\mu_i\exp(l_{qhi})}{\sum_j g^{hard}_{qj}\mu_j\exp(l_{qhj})}.
\]

Retain typed measures exactly once and the native empty-row rules. Source IDs never enter a score numerically.

For Thermal:

\[
 K_i(X,q)=g^{hard}_{qi}(X)k^{raw}_i(X,q)+K_i^{local}(X,q),
 \qquad T(X,q,h)=\sum_iK_i(X,q)h_i.
\]

Every retained nonlocal coefficient has full native amplitude; a gate selects computation, not a new fractional response coefficient. K remains signed and unnormalized over heaters. Routing, context and K remain heat-independent. Zero heat and flow independence, native extraction, surface/material roles and q proxy stay unchanged. A binary source omission can still cause a serious physical error; the binary design removes the attenuation confound, not the need to learn and validate source selection.

Wind retains its own profile-plus-residual transform, aligned coordinates, native units, nonlinear head and local support. No inference wake equation, second rotation, target velocity or special case-ID feature is introduced.

## 4. Learn the cost of the final reads

### 4.1 Exact expected logical work for the declared stochastic model

For independent member Bernoulli variables conditional on the scene and one categorical dispatched seed, the marginal expected final access is:

\[
 \mathbb E[g_{qi}\mid X,q]=P_{qi}+(1-P_{qi})\sum_s\pi_{qs}p_{si}.
\]

This is an expectation over training routing randomness, not a physical probability. The expected number of unique expensive native values is:

\[
 \bar W(X,Q)=\sum_{q,i}\left[P_{qi}+(1-P_{qi})\sum_s\pi_{qs}p_{si}\right].
\]

There is no OR-product saturation from many possible seed paths. Linearity of expectation suffices; final query/source accesses need not be independent. Implement a small Monte Carlo check of this formula on a synthetic graph before using it in training.

Count native unique rows, not attention mass, average permissions, per-head edge counts or physical measure. Physical integration and GPU operations are different measures. Include actual native receiver unions, auxiliary roles and scheduled response calls in telemetry; a primary-Q counter is not complete work.

### 4.2 Budget and loss

Keep every original supervised objective and its TRAIN-fitted scale. Add one explicit work constraint:

\[
 \mathcal L=\mathcal L_{native}+\lambda\,[\bar C-\rho]_+
 +\frac{\zeta}{2}[\bar C-\rho]_+^2,
 \qquad \lambda\leftarrow\operatorname{clip}_{[0,\lambda_{max}]}
 \bigl(\lambda+\eta_\lambda(\bar C-\rho)\bigr).
\]

C is the equal-case expected unique-value fraction, computed on the same TRAIN requests. Use case means before dataset averaging, and retain role/stratum diagnostics. The coefficients and dual-update rule are serialized. This is a proposed constrained-training surrogate; neither exact physical-fidelity constraints nor a guaranteed feasible solution follow from it.

Primary terminal target: retain approximately **60% of removable nonlocal rows**, after allowing the input-defined protection and at least one valid donor. For a query with M>0, an input-only minimum count is \(w_{min,q}=\max(1,|P_q|)\). Use target count \(w_{min,q}+0.60(M-w_{min,q})\), not an impossible fixed fraction on M1 or fully protected rows. Report total retention as well as removable-row retention. Do not call the target an achieved result.

Start from a loose target and anneal it according to §8. Treat the target as a training objective, not a hard inference truncation that overrides learned support. The final deterministic support is measured at its declared threshold. A missed budget is reported as missed, not repaired by top-K pruning at evaluation.

Calibrate the initial work coefficient on at most eight fixed TRAIN batches to give an initial router-gradient scale around 10% of the native task-gradient scale. Use the same small scalar calibration procedure for both datasets: denote the calibrated initial coefficient by lambda0, set zeta=lambda0 and lambda_max=10*lambda0, and update the device-resident dual every10 optimizer macro-steps with eta_lambda=lambda0 using that window's mean budget residual. If the routing task-gradient norm is unresolved, fix that gradient mechanism before coefficient calibration rather than dividing by a numerical zero. Serialize the exact computed lambda0 and all constants. Dual ascent must have room to change support; a coefficient repeatedly pinned to its cap is a reported constraint failure. Do not sweep weights on DEV, secretly reweight Wind roles, or allow a reduced scalar total loss to hide worse supervised terms.

### 4.3 GPU-aware organization without another learned network

The principal hardware-oriented constraint is structural: one shared membership list per native seed, common support across decoder heads, and one dispatched list per query. Queries with the same seed therefore reuse a source-index list and can be evaluated in rectangular blocks. No additional spatial smoothing loss or fixed receiver grid is required in the primary model.

Charge execution padding and direct-protection exceptions separately. At each training review measure deterministic used groups, list widths, query occupancy per group, query/source padding, exceptional pairs, dispatch overhead and dense-backend fraction. Optionally augment the training cost with a **documented smooth padding surrogate** only if actual padding dominates; this is the one reserved executor-aware scientific amendment, not an open regularizer search. It must have a paired continuation identity and cannot silently alter a running lineage.

Do not claim that the logical work formula differentiates measured latency. Actual latency depends on list widths, occupancies, hardware and implementation. Use calibrated cost tables only as engineering evidence; do not relabel their regression as an exact physical or runtime law.

## 5. Gradient design: omitted values must be able to return

### 5.1 Why a selected-only straight-through label is insufficient

If a value is never computed and its permission is excluded by a Boolean mask before normalization, there may be no task derivative indicating that restoring it would help. A cost penalty can then close routes while no useful task signal can reopen them. Round8 must explicitly test this, not just find a nonzero gradient somewhere in the contextualizer.

### 5.2 Dense hard-forward router-learning steps

During designated routing-learning steps, evaluate the original dense native values in bounded tiles, but use the **hard selected field** in the forward pass. Let \(A^{hard}\) be the one-hot query dispatch and \(A^{ST}\) its relaxed straight-through version. Construct:

\[
 g^{ST}=P+(1-P)\odot(A^{ST}B^{ST}).
\]

The forward value of g_ST is the binary deployed/stochastically sampled mask. Its backward path is a surrogate through routing. The value-path derivative is multiplied by the hard forward mask. This adds no second predictor or pooled decoder bank; it deliberately computes omitted values on certain training steps to obtain a useful routing derivative.

For the training-only information reduction do **not** reuse Round7's `where(g>0)` plus log-permission mask, which disconnects zero permissions. Use a stable multiplicative form:

\[
 e_{qhi}=\mu_i\exp(l_{qhi}-m_{qh}),\qquad
 \alpha^{ST}_{qhi}=\frac{g^{ST}_{qi}e_{qhi}}{\sum_j g^{ST}_{qj}e_{qhj}},
\]

with m an eligible-score stabilization and at least the self/protection nonempty guarantee. Invalid records are fixed zero independently of learned membership. Only genuinely empty native scenes use an explicit empty-row branch. This gives a derivative with respect to a zero gate when its hypothetical native value is available. The Thermal scalar reduction uses the same g_ST without attention normalization.

The estimator is biased; do not gradcheck the discontinuous hard decision against finite differences. Gradcheck the continuous relaxation and primitive algebra separately. Input/geometry derivatives exported for inference must use the actual fixed hard route, not this training estimator.

### 5.3 Selected execution steps and training cost disclosure

Use dense hard-forward router learning for the initial routing phase. Later, use it on every fourth optimizer macro-step; the other steps use selected/grouped values and true fixed-support value gradients, plus the explicit differentiable expected-work objective. On those selected steps, acknowledge that omitted task-value derivatives are unavailable. They are restored on the scheduled dense router steps.

The final phase uses deterministic hard memberships and deterministic seed dispatch in its forward prediction, including on dense router-gradient steps. Noise must be annealed away well before the terminal checkpoint. Record deterministic versus stochastic support, disagreement and physical errors at reviews. A stochastic-budget success cannot stand in for deterministic deployment.

Trainable original contextualizer parameters receive both normal task gradients and routing-path gradients. Preserve the original optimization schedule and state. New scalar calibration parameters have their own fresh moments and explicitly named parameter groups. No learned per-layout table, native-ID embedding, frozen teacher network or unrecorded extra optimizer is permitted.

Required native TRAIN mechanism check: identify at least one unprotected omitted source whose finite restoration reduces the current TRAIN loss on a fixed diagnostic batch. With its value path held independent where necessary, show a nonzero surrogate gate gradient in the restorative direction at a sufficiently small gate perturbation. If no beneficial omitted source exists on that batch, record that and use the synthetic fixture for the derivative law; do not search DEV for a showcase. A failed gradient mechanism blocks the router implementation, not the whole native-value idea.

## 6. Grouped native execution

### 6.1 Prepare once per scene

Prepare original token projections and the required MM score capture once. Construct hard native member lists once per seed/scene at evaluation. The primary M consumer does not require ME inspection probabilities in ordinary hot-path calls. Trace-only headwise probabilities and E metadata are lazy, bounded and explicitly requested.

Retain native stores and version binding. Validate scene/weight/normalizer/graph ownership at public boundaries. Do not repeat full tensor-content comparisons, Python parameter traversal, device-to-host scalar conversions or probability reconstruction inside each query tile.

### 6.2 Dispatch, pack, read, restore order

Each query computes existing cheap decoder scores once, chooses its native seed, and records that seed. Group queries by `(scene, seed)` for execution. Sort or compact once per request/chunk with a documented GPU-compatible strategy, not once per query. Native query IDs restore original output order.

For each occupied group, evaluate the rectangular product of its query rows and selected original member records. Use bucketed batched operations or a grouped kernel so a Wind scene with thirty seeds does not create thirty long Python launch sequences. Every query retains its own coordinates, feature vector, final attention weights and nonlinear messages.

Protection exceptions are `(query, source)` records absent from the selected group's list but required by the original local support. Evaluate them once with the identical nonlocal value formula and merge into the same final information normalization and scalar source sum. Do not separately normalize bulk and exception contexts and then add them. Duplicated IDs must never be counted or applied twice.

Source-list and query padding are permitted only as explicit execution overhead. Padding tokens have zero contribution and never become native members. A fixed *tensor tile size* is not a fixed hyperedge count; document the distinction. The number and spatial location of groups still come from actual modules.

### 6.3 Three backends, one scientific field function

1. `dense_hard_reference`: independently implement full native values and binary-mask reductions; this is the correctness and router-learning reference, not a speed claim.
2. `flat_selected_reference`: reuse the Round7 selected primitive where mathematically compatible, with binary support and no fractional scalar attenuation; useful for isolating grouping from model changes.
3. `grouped_native`: the intended efficient executor over native lists and dispatched queries, including protection exceptions and source-order-correct scalar factors.

Backend choice must preserve the selected field. A dense masked fallback for a narrow M or poorly occupied group is allowed and charged; it cannot silently become all-open A2-D. “All-open A2-D” is a separate explicit model mode, exact at the branch boundary, and a diagnostic at adapted weights.

The first GPU implementation can use one request-level synchronization for dynamic shape discovery if unavoidable. Measure and report it. Eliminate repeated per-query/per-seed synchronizations before attempting compiler or custom-kernel work. Do not claim `nonzero` is free on CUDA.

### 6.4 Timing scope and Amdahl limit

Original A2-D preparation and native extraction remain substantial. For attributable disjoint times, if fraction f of the complete call belongs to removable values, even removing all such work cannot beat \(1/(1-f)\). Round7's nested instrumented phase intervals must not be inserted into this formula as a reliable decomposition.

Measure complete host-input to native CPU-output latency, prepared-scene/new-query latency, fixed-query/new-heat reuse, and complete native training steps separately. Use the same hardware, precision, output scope and fair query tiling for all comparators. Do not compare against an artificially tiny-tiled classic as the primary denominator.

Measure B1 and B8 at Q1024/4096/8192 for high-M, and the low-M Q1024/8192 endpoints. B8 repeated-scene shape controls and B8 mixed scenes have different meanings; include at least one mixed-scene training-memory measurement with the real objective, auxiliaries and optimizer. Retain allocated/reserved baselines and peaks. No frozen-context memory claim or smaller-candidate-batch disguise.

## 7. Numerical qualification is separate from research admission

Preserve Round7's strict FP32 misses. Do not retroactively call them passed. PyTorch does not guarantee bitwise identity for algebraically equivalent batched/packed reductions. That fact does not excuse wrong masks, missing biases, measure duplication, reordered source columns, NaNs, or a materially changed native derivative.

Use three comparisons:

- Original all-open dispatch versus the unchanged A2-D source: require the same exact native boundary recovery as before, including one real native optimizer boundary.
- New binary-mask dense reference versus an independent small FP64 literal native-token loop: require close forward, fixed-support gradient and signed-source-law agreement. Suggested data-free tolerances are 1e-10 relative/1e-12 absolute, with a documented scaling adjustment only if the fixture's conditioning requires it before fitting.
- Grouped/flat selected execution versus that dense reference at identical binary masks: record primitive and native per-role absolute, scale-normalized and relative errors, fixed-support VJPs, and the original strict legacy comparison status separately.

Before scientific results, seal a new research-backend numerical envelope using TRAIN-only baseline tiling/reduction repeats and the independent FP64 calculation. Proposed guardrails: normalized primitive RMS discrepancy <=1e-5 of the frozen TRAIN role scale; derived curl/interface proxy <=1e-4 of its frozen role scale; numerical discrepancy also <=1% of same-fixture baseline physical-reference RMSE where that denominator is resolved. Retain max errors and masks, not only RMS. These are new research engineering guardrails, not new physical truth or amendments to historical precision laws. If actual baseline roundoff already exceeds a proposed guard, document the conflict before fitting and keep the grouped backend unqualified rather than enlarging the envelope after seeing trained results.

A failure of the grouped backend means train/evaluate the scientific model with the correct dense hard reference while preserving the failed backend evidence. Do not cancel all learning merely because a faster implementation is not yet qualified. If the *dense reference's mathematical law* fails, repair it before learning. Compiler paths are optional; do not spend the round chasing last-bit compilation parity.

Heat-null, heat-independent flow, source-ID ordering, exact-zero masks, stable empty scenes, units and target isolation remain hard requirements. FP64 prepared increments/adjoints are checked separately from ordinary FP32 endpoint cancellation. Native max-temperature statistics are recomputed at each endpoint and are not linearized by the source sum.

## 8. Bounded scientific training: do not repeat zero-update qualification

### 8.1 Primary pairs and exact inheritance

Use the exact Round7-verified segmented D0 e500 sources. Thermal checkpoint SHA256 is `3ea129d26b890f989ae330369914fcf75b20eba13bb1bd1660eee79f126a5daf`; Wind is `ca7713328f3d4c018ecf91e7cdf2acf6fc5334ae3196c73b54482a5eb028c2d7`. Locate them by the retained lineage receipt and verify actual source/provider bindings. Do not guess paths or substitute H, R-E, mature full-data weights or a different normalizer.

For each dataset branch two independent runs:

| Arm | Parent | New scientific exposure | Purpose |
|---|---|---|---|
| D8 | Own D0 e500, original parameters/moments | e501–2500, 2,000 new epochs | Matched unchanged A2-D continuation |
| G8 | Identical parent, plus new route calibration | Same e501–2500 and case/query streams | Learned budgeted native hyperedge dispatch |

Thermal retains TRAIN150/DEV22, Q1024 primary plus unchanged native auxiliaries and scheduled response, effective/micro 48/24, four optimizer updates including the six-case final macro. Wind retains TRAIN72 direction rows from 24 layouts, DEV24 from eight layouts, Q8192 with the embedded source recipe's 3:1 volume/hub draw, effective/micro 16/8, five updates including the eight-row final macro. Preserve the exact embedded source optimizer, LR schedule, seed and normalization. Do not import the different Round2 five-role Q4096 recipe just because its formal plots are familiar.

The four main continuations contain 36,000 new native optimizer updates, before separately recorded engineering checks. Repeated Wind directions are correlated; layout is the independent grouping unit in descriptive comparisons. Exposed DEV is not TEST.

### 8.2 Primary training schedule

| Absolute age | Forward policy and learning | Budget target |
|---|---|---|
| e500 source; first exact boundary | Explicit original all-open path | No budget penalty; exact migration check |
| e501–700 | Binary stochastic group memberships and categorical query dispatch; dense hard-forward router estimator every step; initially near-open | Anneal expected removable retention from .98 to .85 |
| e701–1500 | Hard selected prediction; dense router-gradient step every fourth macro, selected/grouped physical steps otherwise; anneal stochastic noise to zero by e1500 | Anneal removable retention .85 to .60 |
| e1501–2500 | Deterministic hard forward at every step; dense surrogate-router step every fourth macro, selected/grouped physical steps otherwise | Hold .60, measure actual deterministic retention |

New scalar gates remain trainable through e2500. Relaxation temperatures anneal from 1.0 to 0.5 during the first phase and remain 0.5 for the surrogate; the *forward* is binary throughout. Include exact seed/member noise policies, annealing and RNG namespaces in the recipe. Ensure all queries of the same prepared scene see the same membership sample. Auxiliary calls and query chunks must not secretly resample the scene graph.

Noise annealing changes the sampling distribution, so the simple Bernoulli/categorical expectation is exact only for its specified unit-noise sampling phase. For later scaled/no-noise phases label the sigmoid/softmax budget as a relaxation surrogate; pair it with the actual deterministic hard work and update the budget controller from that observed work. Use the exact stochastic expectation for controller updates through e700; from e701 onward use the deterministic plan's observed unique-work fraction on the same requests, while the differentiable penalty continues to use its explicitly labelled relaxation. Record their gap. Never describe a sigmoid count as an exact expectation for a deterministic threshold. The terminal work target is assessed on actual executed rows, not on that surrogate.

### 8.3 Admission and stopping

Research admission requires: native source availability; all-open baseline recovery; a correct dense binary field reference; finite useful router gradients; target/heat isolation; a bounded real optimizer-memory check; and reserved training/closeout resources. It does **not** require an untrained speedup, an initial sparse histogram, a positive Round7 cost gate, or immediate classic-level fidelity.

Complete at least 1,000 new epochs (absolute e1500) for each healthy admitted pair and normally complete all 2,000. Reviews at e600/e1000/e1500 diagnose learning and resources, not launch a new threshold portfolio. Early total cost or mildly worse fields cannot cancel the comparison. Permissible stops are user request, unresolved semantic/NaN failure, persistent catastrophic native-loss divergence (e.g. >5x own matched D-control over two 100-epoch windows), resource/safety exhaustion, or inability to run the required native objective. A case-specific failed optimized backend is not such a stop when the dense research reference remains correct and affordable.

Keep deterministic field monitors and full history at the original cadence, checkpoints every100 epochs, latest/best-field aliases, loss plots every10 and visible progress. Preserve literal e2500 results as primary; selected-checkpoint results are secondary and use the same prospectively declared monitor. Do not mix graph/timing/field tensors from different ages.

No automatic full-TRAIN or inverse run is authorized by this plan. A potential full-data1000 recipe is an inert deliverable only after the learned development outcome. Formal budget/authorization belongs to the user's next decision.

### 8.4 One bounded amendment allowance, not a new portfolio

If TRAIN measurements show a persistently poor relation between the budget relaxation and deterministic work despite the declared hard-work dual feedback, the allowed correction is a sharper deterministic gate relaxation (temperature0.25 instead of0.5), with an explicit child identity and preserved parent. This is not an evaluation cutoff change or a forced top-K policy. If padding rather than membership dominates, use the one documented padding-cost amendment from §4.3 instead. At most one of these scientific amendments per dataset; no new value architecture or loss-role reweighting. Continue the matched D-control to the same absolute age. All original and amended exposure remains visible.

A second dispatched seed, direct-E consumer, learned prototype, moment bank or wake-cone reader is not the amendment. Those are separately controlled future hypotheses. Do not add them to avoid a negative endpoint.

## 9. Evaluation: three decisions instead of one omnibus pass/fail

### 9.1 Predictor preservation

Use unchanged complete native DEV22 and DEV24 boards at matched e1500/e2500, plus a compact fixed-weight all-selected-TRAIN audit. Preserve every physical component, mean, p90, worst and module-count stratum. Wind near/downstream Ux and vector fidelity must be plainly visible, alongside Uy/Uz, hub, volume and background. Thermal fluid/surface/material T, v/p, curl, q proxy and recomputed module peaks must all remain visible.

Proposed endpoint research bands versus D8: priority means within 5%, priority p90 within 10%; secondary means within10%. A direction with some priority means within10% and no severe tail failure may be retained as a *qualified limitation*, not called preserved accuracy. These bands label the trained result; they are not early training admission conditions. Do not average away a failed field or change the baseline to make the candidate pass.

An adapted all-open mode is a counterfactual at G8 weights. Measure it separately; do not call it a guaranteed accuracy fallback. Historical full-data A2-D and native classics are practical context on the same receivers, not equal-age controls. Reuse valid saved baseline arrays rather than rerunning whole historic campaigns.

### 9.2 Organizer and work

Report deterministic member-count distributions across seeds/layouts, occupied dispatch groups, within-M variation, protected exceptions, actual consumed IDs and the unique nonlinear values evaluated. No graph-size or entropy target is imposed. Dense or singleton groups may be appropriate locally; a universal collapse is a measured outcome to investigate, not an automatic physical interpretation.

At e2500 target at least25% fewer **actually executed** nonlocal rows on high-M scopes after padding/exceptions/dense dispatch. Also report the requested .60 removable budget and how closely both expected/relaxed and actual work track it. A physical-measure-weighted edge count is not a GPU-row count. Only real omitted rows support a sparsity claim.

Same-weight controls on the fixed detail panel: degree-matched direct decoder-score selection, degree-matched geometry selection, member rewiring preserving valid row degree, fixed dispatch with memberships changed, and exactly restored normal execution. At the important fixed queries, show which useful token is retrieved through the dispatched native group despite weak own query score. Recompute proper final attention denominators in every control. These are local reliance/utility controls, not independent proofs of the training effect.

Test query-set independence: a receiver alone, in a permuted request, in a different chunk, and in a mixed batch must get the same deterministic source IDs and mathematical field. Execution buckets may change; the model's selected records must not. Test source permutation with labels permuted consistently and legal geometry changes with state rebuilding. No action uses target fields, errors, actual heat or native IDs as learned inputs.

### 9.3 Execution readiness

The learned field can be promising without a deployable speedup. Separate: (A) correct learned native hypergraph; (B) useful accuracy/work tradeoff; (C) complete-call acceleration. Claim C only with a qualified grouped backend and measured complete-call advantage on declared scopes. A practical target is >=10% reduction on high-M Q8192 B8 or B1, with the other scope and all low-M regressions disclosed; do not require winning every tiny-M call to recognize a scoped benefit.

Preservation and real work savings with latency around parity can support another engineering round. Significant overhead with no useful field/work tradeoff is a negative trained outcome. No speedup is inferred from sparse lists, dense-ST training masks, model parameter count or memory alone.

### 9.4 Spatial seams and inverse interface

Dispatch is discrete. On fixed input-only short lines through natural top-two seed-score crossings, measure output jumps and native finite-difference behavior; compare with D8 and ordinary native spatial variation. Do not impose a new smoothness penalty in the main fit. A good pointwise RMSE with severe routing seams is a material limitation for future geometry optimization.

Replay only the existing Thermal response records and precise heat contracts. No new solver calls, PDE solves, designs or inverse optimization. Keep solver326/326 and all TEST locks. Fixed-support autodiff is not a derivative through a changed dispatch; show that boundary honestly.

## 10. Concrete implementation packages

### W0 — short review and protection

Inspect 782ea56 and later user changes; do not reset. Reuse Round7's resolved source bindings, Wind correction and preserved-parent checks rather than reconstructing tens of thousands of historical files during every iteration. Record active processes and device authority before allocation. Preserve all Round7/Round6 reports and source parameters.

### W1 — new budgeted graph and dispatch API

Add a narrow new module set under the existing native retrieval package, for example `budgeted_graph.py`, `seed_dispatch.py` and `budget_state.py`. These are proposed filenames, not existing source claims. Use a new schema such as `native_budgeted_dispatch_v2`; old v1 loaders stay strict and unchanged.

Implement head-calibrated native membership logits, binary sampling/deterministic membership, exact self/empty/invalid handling, one-seed categorical dispatch, protection union, and the analytical expected final work. Keep E metadata optional. Persist parameters, temperatures, dual variables, phase, RNG identity and budget targets. A future generic typed API may support E, but no unused trainable QE parameters belong to G8.

### W2 — independent dense hard reference and gradient mechanism

Implement binary-mask output independently of the Round7 selected reducer, using original A2-D parameter names and value semantics. Add the dense hard-forward surrogate with multiplicative attention normalization at zero permission. Clearly separate `training_router_estimator` from the exported inference derivative path. Test omitted restoration, selected physical gradients and native heat laws.

### W3 — grouped executor

Reuse original per-token K/V projections and geometry helpers. Prepare source lists per scene/seed; group queries for execution by seed, not by a learned prototype. Add bucketed rectangular value operations and exact protection exceptions. Count padded rows, score/capture work, group launches and host synchronization. Support dense same-mask dispatch for shapes where it is cheaper. Avoid compiling debug traces or hashing entire live tensors inside the hot path.

### W4 — shared TrainingEngine integration

Add a Round8 recipe/factory and a small CLI using the existing engine, source optimizer restoration and native providers. Do not copy another training loop. New launch authorization checks research correctness/resources, not `round7_execution_admission_v1`. Check `fit` with a valid new research receipt actually reaches a native optimizer boundary even when its startup latency record says `not_faster`. Retain negative Round7 rejection tests.

Expose supervised and budget losses separately in logs, along with actual versus relaxed counts, grouped occupancy, deterministic/stochastic mode and exact update/visit counts. Reuse source query namespace and response cadence. One scene graph sample must remain valid across all query tiles/native extraction calls in that objective evaluation.

### W5 — run, review and finish learning

Perform the four admitted e501–2500 continuations with required reviews and scoped resource checks. Budget/memory correctness tests are not substitutes for native fitting. Preserve resumable scientific states before any user/resource stop. Evaluate endpoint fields, graphs and costs from the same weights and deterministic route policy.

### W6 — report and release

Write one new Round8 report, a short evidence index and a reusable guide. Do not revise old results. Commit/push durable implementation, tests, configuration and documentation only; audit the outgoing range and verify remote/local tips. Generated evidence and disposable scripts stay ignored under the mounted data2 root. Cleanup is limited to verified round-owned disposable material.

## 11. Focused tests, not another broad diagnostic campaign

Required test families:

1. Native identity/law: original all-open output and real optimizer recovery, heat independence/linearity, empty/padded types, native measures, signed coefficients, query/source ID order.
2. Budget algebra: Monte Carlo versus expected unique work, no double-counted protection, independent versus deterministic budget labelling, self floor, no reward from fractional attenuation.
3. Learning: isolated MM/query-dispatch gradients, omitted-source restorative derivative with dense surrogate, binary forward equality, correct fixed-support physical gradients, no masked-log zero-gradient mistake.
4. Execution: dense literal versus flat/grouped at identical binary masks, duplicate suppression, source/query permutation, chunk/batch independence, mixed occupancy, padding and exceptions, native forward/backward memory.
5. Lifecycle: exact save/resume of new parameters/dual/noise/phase and original AdamW; stale state rejection after weights/geometry/measures/normalizer change; backward consumption; target poisoning; no TEST access.
6. Launch policy: new research launch can proceed with negative startup speed; old Round7 gate remains immutable; protected GPU and storage rejection precede allocation.

Run affected tests and a small native regression subset. Classify pre-existing failures without spending the whole budget repairing unrelated families. Do not impose bitwise equivalence between a surrogate training gradient and a hard-dispatch finite difference; they intentionally describe different objects.

## 12. Resources and authorization

Use the currently authorized **physical GPU2**, UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, unless the user explicitly changes authority. Round7 records live user jobs on GPU0 and GPU1; do not infer that older permission to use them is still available. Recheck actual process/device state, never kill or move unrelated work. Logical cuda:0 is acceptable only under exact GPU2 mapping.

Proposed bounded envelope: 12 GPU-associated hours, 12 elapsed hours including one reserved closeout hour, 36,000 primary scientific updates plus at most4,000 amendment updates and 200 disposable native engineering updates. Reuse CPU data-free tests freely within the elapsed envelope. Initial engineering should normally take <=2 hours; do not exhaust the round on compiler/parity investigations while a correct reference can train. Reserve the four learning trajectories and endpoint reporting before optional work. If actual throughput exceeds the envelope, stop at a matched checkpoint with a continuation handoff, not a claim of completed maturation.

Use `/home/wanglz/miniconda3/envs/ModularDT/bin/python` after verifying the local interpreter and installed stack. Preserve FP32 and source numerical settings; no framework upgrade, hidden autocast or TF32 change in the main study. Trace compilation costs separately. GPU memory should support useful batch/query work with margin, not be filled for its own sake. Keep original effective batch and query sampling; engineering microbatch changes require same-objective gradient checks and both-arm disclosure.

Store new generated payloads under a verified `/data2/wanglz/ModularDT/diagnostics/<round8-owned-root>` with an ignored workspace symlink. Proposed storage limits: 20GiB retained, 10GiB temporary, 30GiB peak new material and >=20GiB filesystem free headroom. Reuse exact-key shared catalogues; no dataset duplication or historical migration. Record actual availability rather than copying Round7 free-space numbers.

Retain declared scientific milestones, latest/best aliases, source/recipe/checkpoint fingerprints, per-case metrics and selected figure arrays. Delete only validated round-owned disposable caches, failed temporary duplicate payloads and superseded visual exports. Keep compact failure receipts. Recheck active handles and report dependencies before deletion; report files/bytes removed and actual capacity at closeout.

## 13. Report format: make the central answer obvious

The first page should answer: Did a learning run happen? Did native accuracy stay near A2-D? Did deterministic final access shrink? Did grouped execution save real work? Was a complete native call faster? What remains unsuitable for inverse geometry use?

Use no more than five primary figure families, with numerical tables and small selected examples:

- Learning: original native field losses, individual physical priorities, budget/actual support, and progress versus epoch/update/measured time. Explicitly distinguish 500 inherited and 2,000 new epochs.
- Fields: Thermal T/v/p and Wind Ux/Uy/Uz native predictions/errors, shared scales and annotated RMSE/relative L2, plus original Wind wake cuts. Preserve tails and difficult module-count cases.
- Hypergraph: original module seeds, actual native member lists, two or three real query dispatches and the token records they consume. Show bulk, protected exceptions and original local branch separately. Do not plot a node mean as a value object.
- Work/cost: logical unique support, executed/padded rows, group occupancy, complete latency, real objective training memory and same-mask backend comparisons.
- Interventions/boundaries: equal-work controls, membership/dispatch/value removal and exact restoration, a routing seam check, and the limited saved Thermal heat-response replay.

Figures come from saved arrays at exact checkpoint/policy identities. Keep PDF masters and only necessary raster companions locally. Visually inspect labels, native frames, masks and legends; prose captions stay on one source line. Put detailed hashes and numerical failures in the appendix/evidence index, not in every main caption.

The final classification has separate fields: `scientific_learning_completed`, `native_field_preservation`, `learned_work_budget`, `organizer_added_value`, `grouped_backend_numerics`, `complete_latency_gain`, `inverse_interface_scope`. Never use one failed deployment gate to erase a completed useful learning result, or one active graph to override bad fields and cost.

## 14. Explicit stop-list

Do not restart C/H/R-P/R-E/R-G or revive pooled decoder banks. Do not add a low-rank completion, cheap all-source baseline, another utility MLP, fixed H6 slots, a regular ReceiverCover intermediary, target-dependent clustering, an output correction table, a new wake model, or a geometry-dependent heat leak. Do not force a small K at evaluation to meet the budget. Do not treat a shared source list as a shared value tensor. Do not count an unconsumed E graph as a trained direct-E reader. Do not replace native physical-reference fidelity with teacher closeness or attention mass.

Do not authorize a 5k formal run because a tiny TRAIN fixture improved. Do not demand a mature 5k classic be beaten before completing the segmented research fit. Do not rewrite historical checkpoint ages, datasets, numerical tolerances or negative gate decisions. Do not mark a new research goal complete on the basis of dry runs when scientific learning was authorized and remains feasible.

## 15. Sources and rationale boundaries

**Repository evidence.** The Round7 report is the authority for its measured counts, timings, numerical misses and corrected Wind source identity. The pinned code files in §2 establish implemented operations, not independent GPU performance. The prior Round7 plan explains why scientific fitting was blocked. Round3's trained sparse failures remain relevant cautionary evidence, but its completion/utility reader is not the new model.

**External primary sources, used as design precedents rather than guarantees:**

1. Zhu et al., *BiFormer: Vision Transformer with Bi-Level Routing Attention*, CVPR2023. The useful idea is coarse routing followed by fine token interactions in GPU-friendly matrix operations. HONF does not inherit its image-region value summaries or assume its speedups transfer. Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html ; authors' code: https://github.com/rayleizhu/BiFormer .
2. Louizos, Welling and Kingma, *Learning Sparse Neural Networks through L0 Regularization*, ICLR2018, arXiv1712.01312. Motivation for explicit differentiable expected computational support. Round8's binary forward membership and categorical dispatch are specified above; they are not claimed to be the paper's exact hard-concrete operator. https://arxiv.org/abs/1712.01312 .
3. Jang, Gu and Poole, *Categorical Reparameterization with Gumbel-Softmax*, ICLR2017, arXiv1611.01144. Precedent for a categorical relaxation/straight-through training estimator, not exact hard-dispatch geometry derivatives. https://arxiv.org/abs/1611.01144 .
4. Chen et al., *SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer*, CVPR2023. Motivation for structure-aligned sparse execution rather than assuming fewer operations means lower latency. Its window pruning and accuracy/latency numbers are not HONF results. https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html .
5. PyTorch official numerical-accuracy and `torch.nonzero` documentation. Equivalent batched/sliced floating-point calculations need not be bitwise equal; CUDA nonzero entails synchronization. Verify behavior against the installed 2.6 stack before choosing an implementation; do not upgrade the experiment implicitly. https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html ; https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html .

**Final scientific question:** Can original A2-D values remain individually expressive while a learned native hypergraph assigns repeated, budgeted member lists to queries, giving a useful fidelity–executed-work tradeoff? Round8 is designed to answer that question with actual training. Speed and physical inverse validity remain subsequent measured outcomes, not assumptions needed to start.
