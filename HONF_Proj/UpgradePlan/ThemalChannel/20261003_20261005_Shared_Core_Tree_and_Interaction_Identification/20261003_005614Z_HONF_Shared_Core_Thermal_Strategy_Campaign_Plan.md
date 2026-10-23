# HONF shared-core alignment and ThermalChannel strategy campaign

## Goal-mode assignment

### Execution scope amended on 2026-10-04

The user instructed: "I want you to draw conclusion from 500 epoch results,
no longer wait for 1k epoch." All five arms completed 500 genuine full-dataset
epochs. The final common comparison therefore uses exact 500 and the separately
preserved, predeclared field selections within 500. The original requirement
to finish additional H1000 models and the subsequent paired-head training are
superseded for this delivery; no further training is launched. Completed
Native/Fine/Local 1000 results remain supplementary evidence, not substitutes
for the common 500 cohort.

The active Tree continuation was stopped at the user's request after 666
complete epochs. Its finished 500 parent, later checkpoints, failed/recovered
lineages and saved numerical evidence are preserved. Finish the inspected
Predictor/Organizer/Inverse report and tested exploratory manual 5,000-epoch recipes
from the actual reviewed parent stage. Do not launch 5,000. The original design
and ladder below remain an archival record of the earlier assignment.

**Research goal:** learn a hypergraph-inspired organizer through multi-field reconstruction, so that it represents useful module–environment interactions and can later be reused in flexible inverse design. A smaller integer K, a good average field score, and an executable inverse sampler are different achievements. This campaign must evaluate their connection.

**This is a model-development and staged-training campaign, not another diagnosis-only round.** Implement and compare several strategies, screen them at 100 genuine epochs, extend several to 500, and finish 2–3 new candidate models at 1,000 epochs. Deliver executable, manually launched 5,000-epoch recipes for 1–2 selected candidates. Do not start those 5,000-epoch runs.

Use `agent/honf-core-next`. The reviewed scientific implementation is `0076b2898d47b1d12a87e89ff10f5005b12543b5`; the inspected subsequent tip, `0d1f57c2d8ad0922de7b0b835e5de138bdb1e6f5`, changes report figure links. Inspect the actual current checkout before editing. Preserve unrelated newer work.

The user's present epoch-based authorization supersedes the earlier six/eight-hour diagnostic-round limits. It does **not** authorize unlimited architectural sweeps, new CFD generation, or automatic 5,000-epoch training. Use the finite run portfolio and staged resource forecast below. GPUs **1 and 2** are authorized. Pause new WindFarm scientific training; retain WindFarm as a shared-core compatibility test.

### Mandatory end products

1. A measured WindFarm/ThermalChannel core-alignment report and reusable alignment tests.
2. New shared-core model implementations, not dataset-local copies of interaction algebra.
3. Actual 100-epoch screening results for the portfolio; several completed 500-epoch models; **2–3 distinct new candidates completed at 1,000 epochs** unless a documented, genuinely unresolved execution failure makes this impossible.
4. Forward fidelity, graph quality, training/inference cost, and inverse-use evaluation at the selected versions.
5. A concise manual 5k launch guide and runnable launch configurations for 1–2 candidates, with both continuation and fresh-run semantics made explicit.
6. A final English report opening with an accessible **Predictor / Organizer / Inverse** gain–miss summary and containing inspected measured visualizations.

A simulation-free smoke test, a fitted selector, an available checkpoint, or a short adaptation pilot is not a substitute for these training outcomes. Conversely, do not manufacture successful graph or inverse claims merely to meet a deliverable.

---

## 1. What the latest evidence supports

The sources below are the basis for the decisions, not evidence that the proposed candidates will succeed.

**Predictor:** the latest Wind full-access co-adaptation improves volume RMSE from 0.008363 to 0.007938 m/s. Its signed-direction sparse organizer does not consistently beat the summary control. Thermal value8 beats value4 in 32/32 exposed absolute calibration cells, but only 2/32 beat Run1804; all reported channel-average absolute errors remain worse than Run1804. Thus the useful base is the retained native model and training infrastructure, not a presumption that the latest modified weights are the best initialization. [S1, Predictor; Thermal predictor]

**Organizer:** measured adequate sparse actions exist in 12/12 latest primary Wind layouts, but conservative selection falls back in 10/12. Learned source selection wins all roles in 0/50 primary comparisons against a degree/weight-matched geometry control. A K2 partition does beat a K1 partition at the same work in one layout, so grouping is not categorically useless. Its broad advantage remains unproven. [S1, Availability; Group utility]

**Current split-learning scope:** the inspected `directional_packets.py` returns zero split logits and defaults to the root when no explicit cut is supplied. Its trained residual scores MM sources; the other typed routes remain full, and an external action-selection procedure supplies the evaluated cuts. This is not a directly trained end-to-end split policy. It does not rule out useful grouping; it limits what that particular experiment tested. H-tree below deliberately changes that learning problem. [S9]

**Specific selector defect:** a source-empty packet enters an unmasked pooled maximum, producing an input roughly 1,345 training standard deviations from the fit support and an absurd predicted log risk. Fix empty-set semantics in reusable feature reductions. Do not hide this by clipping the development outcome, relaxing a safety margin, or fitting on that outcome. [S1, layout168 audit]

**Inverse:** the velocity formulation removes the historical corner-saturation numerical mechanism. Eight finite trajectories still yield six native-support failures and negligible response to changed observations. A stable sampler is not yet a useful conditional inverse model. [S1, Inverse]

**Workflow:** the previous experiments count optimizer updates, partial family cycles, and epochs differently. The current campaign uses the native full-case epoch definition. It is not a continuation of the eight-family response-adaptation counter under a new name.

### Decisions for this campaign

- Retain **Run1804 selected e4738** as the frozen, mature scientific reference.
- Train new candidate cores from a matched fresh global initialization. Keep the same existing Stage-A local thermal surrogate attached/frozen in every main arm.
- Do not initialize the main campaign from Thermal G-u1300/value8-u1540. Their already-drifted weights would confound architectural comparison.
- Separate a **fresh Run1804-family Dense control** from the **mature Run1804 reference**.
- Include a **three-term full-access control** to isolate the cost of removing coarse/local interaction bypasses. A failure of that control must not be assigned to hypergraph selection.
- Learn the organization inside the reconstruction model. A post-hoc absolute-risk regressor that normally chooses Dense is not the main new organizer.
- Do not make an early 10%-of-mature-Dense rule a blocking prerequisite for continuing a healthy 100-epoch candidate.
- Do not force K variance or a preferred K histogram.

---

## 2. Inter-model alignment: same reusable mathematics, different physical interfaces

### 2.1 What the inspected code actually does

Both `WindFarmForwardModel` and `ChannelThermalHONFModel` use `honf_forward_core.interface_fields.InterfaceFieldCore` for their interface-field architectures. That is a real shared implementation. It does **not** imply that all their selected architecture identifiers execute the same mathematical network. [S2–S4]

`InterfaceFieldCore` selects `SharedInterfaceContext` for `dense_pairwise_field`, retaining coarse and local context paths. It selects `ThreeTermInterfaceContext` for the group-control families and `three_term_full_access_honf`. The latter implements a background term and the fine module/environment terms without the historical coarse bank or local-neighbor field branch. [S4–S5]

Wind's wrapper additionally has a fixed architecture allowlist and a historical sparse-incidence workaround that constructs a 2-D configuration before restoring 3-D settings. Thermal has a legitimate P0/P1/P2 local-surrogate coupling loop, case-specific target normalization, and thermal outputs that Wind does not have. [S2–S3]

**Do not claim alignment from class names alone. Do not remove legitimate Thermal physics to make the wrappers look alike.**

### 2.2 Deliver a small architecture map

Record, for each retained reference and each new candidate:

- wrapper and input-adapter class;
- shared-core class and architecture identifier;
- encoder dimensions, coordinate convention, field channels, and source measures;
- MM/ME/EM preparation, QM/QE decoding, and each active coarse/global/local path;
- organizer construction time and phase semantics;
- local-surrogate ownership and freeze status;
- target normalization and output reconstruction;
- trainable parameter counts by component;
- the model-side interface exported to inverse code.

Use a simple table plus a call/data-flow diagram. This is descriptive metadata, not a new hash/approval framework.

### 2.3 Shared-code requirements

All new learned grouping, source-to-group reduction, group control, access algebra, gate sampling, and fine-reader changes live under `HONF_Proj/src/honf_forward_core/`.

Case adapters supply only:

- physical source features and typed validity;
- coordinates, reference frame, source/receiver measures, and dimensionality;
- prescribed operating conditions and boundary features;
- optional module characteristic lengths;
- receiver roles and physical-output transforms;
- Thermal's local solver/coupling wrapper.

Do not put `if case == ThermalChannel` branches inside the new interaction equations. Dataset-specific losses and known generator controls remain in the case package.

Register new architectures and supported spatial dimensions centrally. Keep historical checkpoint loading intact. For the new profiles, replace the need for Wind's 2-D-then-3-D construction workaround with explicit supported-dimension validation; retain the historical loader path for old checkpoints.

### 2.4 Minimal alignment tests before the screening launches

Execute rather than merely inspect:

1. Native retained Thermal and Wind checkpoint replay after the refactor, without a new policy: unchanged outputs to declared numerical tolerance.
2. One genuine optimizer step and one prepared multi-chunk decode through each wrapper using the relevant shared-core implementation. Use disposable weights; do not alter historical models.
3. New organizer in 2-D and 3-D: module permutation, padded-source masking, query order/chunk invariance, phase/state freshness, empty-source-type behavior, and nonzero task gradients.
4. Same canonical encoded example through the two core-call adapters: compare common latent/read quantities when dimensions and inputs are intentionally identical. Do **not** require Thermal temperatures and Wind velocities to agree.
5. New geometry feature padding preserves axis identity. Test the 2-D embedding in 3-D at the organizer feature/action level where the mathematical equivalence applies; do not claim whole-checkpoint dimensional invariance across unrelated learned encoders.
6. Finalist architecture construction and Q-small forward/backward on a Wind input, using disposable new weights. This is compatibility, not Wind transfer accuracy.

Bound this alignment pass. Reuse focused tests, repair ordinary issues, and then train. Do not rewrite every historical architecture or require all archived diagnostics to run before proceeding.

---

## 3. Portfolio: two controls and three substantive candidate strategies

Use descriptive aliases below; allocate actual run IDs from the repository to avoid collisions.

| Alias | Model | Scientific question |
|---|---|---|
| B-native | Fresh native Dense, Run1804-family assembly | How much accuracy and training behavior comes from the established full model at the same new training age? |
| B-fine | Fresh three-term full-access fine-interaction model | Can the common no-bypass spine learn the task adequately without adaptive grouping? |
| H-tree | End-to-end adaptive receiver hierarchy with fine-source reads | Can reconstruction directly learn where receiver groups need different source information? |
| H-overlap | Input-conditioned overlapping source-control hyperedges | Does collective source organization work better without a disjoint receiver-tree restriction? |
| H-local | H-overlap with explicit near-interface access and adaptive farther interactions | Does protecting local interactions remove the recurrent near-field/port failure while retaining useful nonlocal groups? |

The mature Run1804 checkpoint is an additional fixed reference, not one of the fresh arms. Existing Run1502 and selected historical models may be replayed on the final comparison panel; do not retrain all historical models.

### What is shared

H-tree/H-overlap/H-local and B-fine share:

- the same local physical encoders, hidden width, fine message/update/read kernels, field head and field assembly;
- the same local thermal model, predicted-port coupling and output normalizers;
- the same physical training cases, epoch/query stream, batch semantics, and primary value losses;
- a typed organization API and the same source-preserving read algebra below.

Use the native profile's initial dimensions unless a real memory issue requires a documented common change: currently the checked-in Dense profile has hidden width 256, message width 128, four attention heads, relative Fourier order four, and 192 Thermal environment tokens. The local case profile requests batch size 48 and 1,024 sampled field points per case. Confirm the archived Run1804 configuration rather than assuming the current profile exactly reproduces it. [S6–S7]

### What is deliberately different

B-native retains the established coarse/local contexts. B-fine and the H candidates use the explicit three-term assembly from initialization. This is **not** the previous destructive conversion of a trained Dense head. The new architecture is trained from the beginning, and B-fine measures whether its capacity is sufficient.

The three candidates differ in organizing hypothesis, not in arbitrary learning rates, channel losses, hidden widths, or access to target information. Do not give one candidate extra response labels without recording a distinct training-strategy variant.

---

## 4. Shared mathematical interface

Let `d` be a design, `c` its operating/boundary context, and `p` the physical coupling phase. Fine module and environment sources retain states `z_s`, coordinates `x_s`, positive measures `mu_s`, and validity masks. A physical receiver `r` may be a module, environment node, port, or field query.

For each mechanism `tau in {MM, ME, EM, QM, QE}`, the organizer returns:

- admitted group identities and source memberships `b_es^tau >= 0`;
- receiver access `a_re^tau >= 0`, normalized over valid active groups;
- low-dimensional collective controls `h_e^tau`;
- an explicit group/plan state independent of the requested query batch;
- fine-source eligibility, phase, and the declared fallback/near-access information.

### 4.1 Build group controls without replacing fine values

Compute module and environmental summaries **separately**:

`m_e^S = sum_{s in S} mu_s b_es`

`zbar_e^S = sum_{s in S} mu_s b_es phi_S(z_s) / (m_e^S + eps)`

`h_e^tau = Psi_tau(zbar_e^M, zbar_e^E, log(1+m_e^M), log(1+m_e^E), receiver_summary_e, c, p)`.

Use presence flags for empty source types. A zero-mass type contributes a masked zero summary, not NaNs or a fabricated physical source. Shared nonlinear `Psi` permits collective dependence on several members. That is a model mechanism, not proof of a fundamental higher-order physical law.

The original `z_s`, source coordinates, quadrature, projected keys/values, and physical source identities remain available to the fine reader. Never replace the entire environmental field read by `D(h_e)` alone.

Default control width is 16; record changes. Initialize control-output modulation near identity, with the last gain projection zero, so a new control does not cause a gratuitous amplitude jump at the first step.

### 4.2 Deduplicate paths before physical attention

Define the access density and source-resolved control moment:

`rho_rs = sum_e a_re b_es`

`n_rs = sum_e a_re b_es h_e`

`hbar_rs = n_rs / (rho_rs + eps)` on positive support, zero otherwise.

In mathematical notation, the central source-preserving operation is

\[
\rho_{rs}=\sum_e a_{re}b_{es},\qquad
 n_{rs}=\sum_e a_{re}b_{es}h_e,\qquad
 \bar h_{rs}=\frac{n_{rs}}{\rho_{rs}+\varepsilon}.
\]

There is one physical `(receiver,source)` read, even if multiple groups expose the same source. For source-type `S`, use either the inherited native normalization when exactly applicable or the following explicitly implemented normalized access density:

`rho_mean_r = sum_s mu_s rho_rs / sum_s mu_s`

`w_rs = rho_rs / rho_mean_r` when its denominator is positive.

This normalization makes source-uniform positive access equivalent to `w=1` and avoids an artificial `1/K` amplitude effect. It is part of the **new** model specification, not a claim of arbitrary compatibility with every historical kappa convention. Do not silently copy Run1502's kappa or multiplicity factors into a different normalization.

For an additive fine message:

`M_r = [sum_s mu_s w_rs Gamma_tau(m_theta(z_r,z_s,relative_rs), hbar_rs)] / Z_native_r`.

`Z_native_r` must be the explicitly audited all-access denominator for this native mechanism; do not mix a selected-source denominator from one implementation with an all-source numerator from another. Full-access/control-identity parity with B-fine is a required numerical test.

For environmental attention, retain exactly one source-union softmax:

`A_rsh = softmax_s( dot(Q_rh,K_sh)/sqrt(d_h) + geom_rsh + log(mu_s w_rs) + control_score(hbar_rs) )`

`C_E(r) = output_projection(sum_s A_rs * control_value(V_s,hbar_rs))`.

Use true masked support for zero permissions. No separate per-group environmental softmax followed by averaging. Final latent field composition remains:

`C(q) = C_global(q,c) + C_M(q) + C_E(q)`

`u_hat(q) = D_theta(LayerNorm(C(q)))`.

The global background may use declared operating/boundary quantities. Inventory any layout summaries it receives. Do not feed it a hidden dense source-interaction bank that reconstructs the same physics around the organizer.

### 4.3 Preparation and physical phases

Generate plans from current pre-interaction encoded sources and prescribed context, not target fields or Dense predictions. Apply the plan to the relevant MM/ME/EM aggregation **before** unrestricted cross-system mixing. Query groups alone are insufficient for a claim about physical source dependence if the selected tokens were already globally mixed.

Use the same organizer parameters across P0/P1/P2 with a phase/context input where appropriate. Recompute the continuous physical state at each native phase; do not reuse stale P0 data at P1/P2. Wind supplies its one field-preparation phase. State explicitly which paths are organized and which remain all-access.

Initially use the B-fine shared nonlinear update kernels. Do not unfreeze/retrain the local Stage-A thermal model in this primary architecture comparison. If later evidence identifies a local-model issue, make that a separately identified strategy revision.

### 4.4 What K means

Report separately:

- allocated candidate capacity;
- admitted groups;
- source-bearing, nonredundant case-local groups;
- query-local participating groups;
- unique physical source–receiver pairs;
- actual executed rows and calls.

Define case K from the input-only receiver/source catalogue, not the accidentally requested Q512 batch. Canonicalize identical actions only when equality of the implemented operator is established; otherwise report near-duplicates without claiming exact compaction. Never count full-access fallback as learned sparse organization.

---

## 5. Strategy H-tree: directly learned adaptive receiver hierarchy

### 5.1 Structure

Use a bounded physical receiver tree, initially depth three (at most eight leaves), with smooth receiver overlap and input-relative geometry. Build typed receiver catalogues: module receivers for MM/ME, environment receivers for EM, and physical input-derived field/port anchors for query access. A module tree must not be split primarily by thousands of unrelated environment anchors.

Each internal node predicts a split logit from signed geometry, weighted receiver states, source summaries, phase, and prescribed context. Preserve spatial-axis identity and distinguish absent axes explicitly. A geometric hierarchy is an index; its learned source sets and uses supply the substantive hypothesis.

Let `s_n in [0,1]` be a split value. Starting with receiver mass `t_root(r)=1`, recursively define:

`a_rn = t_n(r) * (1-s_n)` for an internal node,

`t_left(r) = t_n(r) * s_n * beta_left(r)`

`t_right(r) = t_n(r) * s_n * beta_right(r)`

with `beta_left + beta_right = 1`; leaves consume all arriving mass. Thus the active receiver weights sum to one without an arbitrary output rescaling.

Hard evaluation uses deterministic split decisions and a real frontier. Source memberships at each selected node use a geometry prior plus a learned residual and a sparse normalization over eligible sources. Keep module and environment source masses separate.

### 5.2 Training path

Exercise parent and child computations during the initial training period. A split whose children initially copy the same source permissions must preserve its parent action. Subsequently let the children learn different supports and controls. An exact copy is an initialization check, not a useful-grouping result.

From epoch 26, mix scheduled frontier exercises with learned frontiers. By epoch 100, most training calls and every main validation call use the deterministic learned frontier; retain a small recorded exploration fraction to keep unselected branches trainable. Suggested learned/exploration fractions are 0.75/0.25 after epoch 100 and 0.90/0.10 after epoch 300.

The split policy is trained by reconstruction and the structural objective, with the hard/soft gradient mechanism below. Do not require a separate ridge head to predict absolute CFD errors before the model may choose a frontier.

### 5.3 Main test

At matched source work, does a learned split outperform its parent action and a geometry-only split? Does that behavior transfer to another layout of the same M or another observed operating context? Requested K alone does not answer either question.

---

## 6. Strategy H-overlap: adaptive overlapping source-control hyperedges

### 6.1 Structure

Remove the receiver-partition restriction. Use a bounded bank of **eight** provisional edge proposals initially; this is `K_max`, not a target or fixed active K. Proposals combine learned feature queries with input-derived physical centroids/extents. They are not module-ID embeddings or output-field clusters.

For each proposal, calculate typed source scores from local source states, signed relative geometry, and prescribed context. Normalize scores sparsely over eligible sources for each source type, for example with the already supported sparsemax operator. Receiver-to-edge access is separately normalized over admitted edges.

Use separate learned **case/phase admission gates** `z_e`, not an observed-q histogram threshold. Source membership, receiver access, and admission jointly define a group. Gates are conditional on current input features and group summaries, not on model error at inference.

### 6.2 Exact admission and usable gradients

Reuse the existing tested stochastic-gate primitives rather than implementing an opaque new estimator. A concrete option is a stretched hard-concrete gate:

`v = sigmoid((log(u)-log(1-u)+eta_e)/temperature)`

`z_e = clip(v*(upper-lower)+lower, 0, 1)`.

Inference uses a documented deterministic estimator of the same gate. Retain at least one admissible group through a deterministic highest-score rescue if all gates close; count and report that rescue. It does not constitute successful sparsification. Do not introduce a minimum K greater than one.

Gate closure removes an access/control function, not the stored physical source. A remaining group can still expose a source if the task needs it. In the K1/all-source limit, pair work stays dense and its structural cost must say so. This prevents the old mistake of rewarding a universal root merely because its label count is small.

Source and query normalization remain live inside the chosen active set. Include original fine source measures and the shared fine read; do not average away source-local states. Handle empty typed source sets and exact gate boundaries explicitly.

### 6.3 Main test

Do overlapping groups explain useful collective source roles better than the tree, at similar physical error and work? Test disabling only group controls while preserving access, and changing access while retaining controls. This separates a useful group state from a useful selection mask.

AllSet supplies precedent for trainable source-to-group/group-to-receiver multiset computations; sparsemax supplies an exact sparse normalization; hard-concrete supplies a trainable discrete-structure mechanism. None provides a guarantee of physical identifiability or useful adaptive K in HONF. [M1–M3]

---

## 7. Strategy H-local: preserve local interactions, organize farther information

Use H-overlap's learnable groups and parameters, but explicitly preserve a near-interaction region.

For each typed source–receiver pair, the adapter supplies a characteristic physical length from module radius/extent or a positive environment quadrature length. Define a fixed, documented smooth local-access envelope `n_rs in [0,1]`: full within one declared local scale and tapered to zero by a second scale. Initial choices should follow the existing physical neighborhood definitions and be identical across training and evaluation, not fitted to development outcomes.

Use:

`w_rs_hybrid = n_rs + (1-n_rs) * w_rs_far`.

For attention, normalize once over the combined fine source set. Do not compute independent near and far softmaxes and add them. For additive messages, preserve the same native measure/denominator convention as the other candidates.

Scale collective far-control modulation by `(1-n_rs)` for the initial experiment, while local fine messages use their ordinary physical kernels. Thus the near path is explicit, source-resolved and included in the cost—not a high-capacity unreported bypass.

The learned count refers to far-interaction groups; the near envelope is a separate deterministic component and must be drawn/reported separately. Do not call it learned adaptive K.

**Hypothesis:** several past models damaged near-interface and port behavior by changing local information access while chasing an aggregate field objective. This candidate tests whether local protection allows useful nonlocal organization without that tradeoff. It is not justified to call every far interaction redundant or every near interaction physically necessary.

Measure total work, far work, and the share made mandatory by local protection. A negligible far region or an all-access near region is an informative negative result, not a speedup.

---

## 8. Optimization common to the candidates

### 8.1 Preserve actual hard values and learn omitted information

The historical log-support mask can block the local restoration gradient of an omitted QE source. Reuse the corrected hard-value/soft-organizer construction where needed:

`y_h = F_theta(d,c; hard_plan.detach())`

`y_s = F_stopgrad(theta)(d,c; soft_plan_phi)`

`y_train = y_h + y_s - stopgrad(y_s)`.

The forward value is hard. Physical-parameter gradients come from the hard model. Organizer gradients come through a continuous shadow, including presently omitted sources. Detach the shadow's physical encoders, kernels, and relevant input feature paths so it does not secretly become a second physical gradient.

The shadow must actually expose a restoration derivative. If the hard path uses sparsemax source memberships or clipped admission gates, do not simply reuse the same exact zeros inside the shadow's log-support operation. Use positive, temperature-controlled softmax/sigmoid permissions on eligible sources and untruncated gate probabilities for this training-only path, with the same physical validity masks. The reported/inference operator remains the hard one. Verify a locally omitted QE source's gradient against a finite restore intervention on a real training example. This is an explicitly biased training estimator, not an identity of the soft and hard operators.

A memory-efficient implementation can backpropagate the hard loss, retain its detached output adjoint, then evaluate a separate soft VJP for organizer parameters. Match it against the explicit expression on a small case. Report the extra forward/backward cost.

Test all-access shadow identity, zero optimizer change for intentionally frozen components, gradient reset between updates, checkpoint-recomputation randomness, inactive modules, no-eligible-pair MM, and actual finite restoration directions. Do not claim an exact derivative of a discrete topology.

### 8.2 Value-first reconstruction objective

Primary value losses are the maintained native field, interface, material-temperature, predicted-port and consistency losses, with the same settings across the five fresh arms. Do not inherit the latest value4/value8 adaptation objective as though it were the original full-dataset objective.

Use training-only channel scales and report physical units separately. Keep pressure, field temperature, surface temperature, outside temperature, material temperature and `q_normal` distinct. They are not interchangeable supervisory quantities.

Enable new hypergraph structural terms explicitly; leave the historical `target_active_edges=3` regularizer disabled. An unchanged configuration default must not silently impose a preferred K.

### 8.3 Penalize information work, not K alone

Use a bounded structural proxy with separately reported components:

`C = mean_tau(C_unique_pairs_tau) + lambda_inc*C_source_group_incidence + lambda_group*C_active_groups`.

Normalize each mechanism against its full eligible work so enormous QE tables do not erase module-transport learning. Also report total unnormalized work; the balanced objective is not a latency estimator. Keep `lambda_group` small relative to the pair/incidence term. One giant all-source edge pays for dense work.

Use smooth occupancy only for the training surrogate and exact positive support for hard measurements. A soft probability is not a skipped operation. Suggested starting incidence/group coefficients are 0.05/0.01 on their normalized terms; these are hypotheses, not calibrated physical constants.

Ramp structural pressure over epochs 26–100. Set its overall scale from several train-only batches spanning M, using a modest initial structural/task organizer-gradient ratio (approximately 2%, capped if the task gradient is near zero). Do not derive a huge coefficient by dividing by an absent task gradient. Log the scale and actual gradient ratios. One bounded coefficient revision is permitted per candidate before epoch 100; after that it defines a versioned training-policy amendment.

There is no fixed target K or required variance. When structural pressure causes a sustained field collapse, reduce or pause that pressure and train the actual physical model; do not silently restore all access and call that the same sparse result.

### 8.4 Decision-response exposure without another atlas-dominated refit

Epochs1–100 are the common multi-field task. Starting after epoch 100, add a **small** common auxiliary finite-response task to all surviving fresh controls/candidates, using only the established training-family records.

Use one baseline/trial pair per epoch initially, cycling physical families and perturbation types; do not evaluate an eleven-state stencil inside every historical minibatch. Calculate both predictions with the same absolute model and comparable aligned queries. Count these extra examples/optimizer work separately from the full-case epoch.

Suggested objective:

`L = L_native_values + lambda_delta L_aligned_delta + lambda_peak L_per_module_peak_delta + lambda_p L_pressure_delta + lambda_C C`.

A combined auxiliary gradient around 5% of the value gradient is a starting calibration, measured across several training families rather than one. Ramp over epochs 101–200. Preserve broad value data every epoch; reduce auxiliary influence if the matched dense control and H models both forget values. No automatic universal multiplier based on a tiny response denominator.

For absent numerical response floors, use absolute error scaled by a declared train quantity and show the reference response magnitude separately. Do not report reliable response-relative accuracy or null physics from an unresolved denominator. `q_normal` remains a generator proxy. Known fixed-geometry heat nulls can be used only as explicitly Thermal-specific controls supported by this generator.

If a response-aware variant is tried on only one model, create a matched value-only sibling from the same parent and label it a training-strategy experiment. Do not attribute a loss change to architecture.

---

## 9. Data, initialization and a genuine epoch

### 9.1 Primary data

Use the complete available native Thermal training split (currently 600 packed records, subject to actual dataset verification). One epoch is one complete pass over that fixed split, with all selected cases visited once through the native bucketed loader. New random field queries may be drawn each epoch.

Set `max_train_batches_per_epoch=null` for every formal run. A smoke limited to a few batches is not an epoch in the campaign table. Do not use the eight response families as the entire forward-training dataset.

The existing 90-row `test` split has repeatedly been used as development. Use the 89-case sensitivity excluding the cross-split physical duplicate 0273 as the main new aggregate and retain the 90-row compatibility table. Do not invent a fresh test label for previously exposed families. Existing unseen-to-a-specific-student challenges can be reported with that narrower description.

Normalize from training data only; use a shared canonical physical-unit evaluator across checkpoints. A model's own target normalization must not make cross-model error comparisons incomparable.

### 9.2 Initialization

Train the fresh global models from seed 0; embed the same pre-existing Stage-A local model. Materialize lazy modules first. Explicitly copy compatible **initial** shared tensors from a canonical B-fine initialization into H-tree/H-overlap/H-local; use independent controlled RNG streams for their added modules. This equalizes shared initial physical tensors despite different constructor RNG consumption.

B-native uses the same seed, dataset, native initializer conventions, local model and task. It has different context parameters by design. The mature Run1804 is evaluation-only.

Do not load mature Run1804/G/value8 weights into a main candidate and call its next 100 epochs a cold-start matched comparison. A warm-start alternative is allowed only as a separately identified strategy with total pretraining exposure disclosed.

### 9.3 Batch and optimizer

Begin with effective batch 48, Q1024 field queries and the native dynamic module-count bucketing. Prefer microbatching and gradient accumulation when memory is constrained; weight every microbatch by its actual case contribution. Preserve the full epoch and effective batch rather than dropping high-M cases.

Use the archived/native optimizer as a starting point (the checked-in profile uses Adam-style training at 3e-4, weight decay 1e-5, FP32, gradient clipping 1). Verify the exact implementation. Train all intended new global/organizer components, not only the final linear layers. Keep only the Stage-A local model frozen in the primary fresh comparison.

Separate organizer and physical optimizer groups if required, but record both actual steps and parameter coverage. An organizer created after optimizer construction must not be silently absent from optimization.

Use an absolute-epoch learning-rate/structure schedule that does not change merely because a job stops at 100 or 500. A stable plateau-based LR schedule is also acceptable if its state is fully preserved and its decisions are documented. Do not reset a cosine horizon or warm-up at every resumed stage. Do not cool the LR to essentially zero at epoch 100 and then call the resumed run a fair 500-epoch continuation.

---

## 10. Staged campaign and model selection

### 10.1 Default run ladder

| Stage | Runs | Required work | Review purpose |
|---|---|---|---|
| Integration | Five profiles; tiny disposable cases | real forward/backward and short no-cap loader checks | expose plumbing, masking and cost issues |
| Screening | B-native, B-fine, H-tree, H-overlap, H-local | **100 complete epochs each** | establish fitting and nontrivial structure; identify broken hypotheses |
| Development | both fresh controls plus 2–3 H candidates; normally all three unless clearly dominated/broken | **continue to 500 epochs** | compare learning curves, task/graph tradeoffs, and early inverse-readiness |
| Final candidates | **2–3 new H candidate versions** | **continue to 1,000 epochs** | substantive trained comparison; exact and selected checkpoints |
| Matched controls | B-native and B-fine | continue to 1,000 unless a matching archived run is demonstrably equivalent in data, task, init, schedule and exposure | avoid comparing young candidates only against a mature reference |
| Handoff | 1–2 selected candidates | prepare manual 5k recipes only | user decides launch |

Default: three H candidates reach500 and two reach1,000; use the third finalist when its different hypothesis remains plausible or complementary. Do not count B-native/B-fine as satisfying the user's 2–3 new-finalist requirement.

Allow up to **two repaired/alternative 100-epoch screens** in addition to the initial five. Avoid an open-ended Cartesian product of architectures, seeds, losses and capacities. If one main candidate cannot execute reliably, repair/replace it early enough to obtain two real final candidates.

### 10.2 At 100 epochs

Review complete training coverage, actual hard forward fidelity, group/source support, parameter gradients and source-role controls. An early candidate need not beat mature Run1804 to continue. Use trends and same-age B-native/B-fine, not only a single endpoint ranking.

Stop or redesign for a concrete failure: disconnected organizer, target leakage, severe persistent nonfinite behavior, constant unintended all-access dispatch, or an identified normalization error. Ordinary modest accuracy deficits and no speedup are not sufficient reasons to avoid the requested longer experiment.

A zero or full K outcome is a finding, not automatically a programming failure. Test whether its interventions affect predictions before deciding why it happened.

### 10.3 At 500 epochs

Evaluate the full development field/port/material population, existing finite-response families, group utility, query/grid robustness, and an inverse-conditioning readiness panel. Choose 2–3 finalists using a Pareto assessment of:

- value and interface/decision fidelity, including tails;
- evidence that the organizer changes consequential physical computation;
- adaptive structure/source support without forced diversity;
- optimization stability and measured cost;
- learning trend, not only the best one-time score.

Preserve one architecture with a distinctly promising organizer even if its average error is not the smallest, provided its physical deficit remains a credible research tradeoff. Do not promote a visually impressive graph whose field prediction is unusable.

### 10.4 At 1,000 epochs

Finish the selected runs. Report **exact e1000** and separately the predeclared selected checkpoint within 1,000. Evaluate every quantity at the same selected weights; do not assemble a synthetic winner from one model's field-best, another's temperature-best, and a third's graph snapshot.

The absence of mature performance can be reported honestly after this training. It must not be converted into a claim that all possible HONF organizations fail. Conversely, longer training does not certify physical interpretability.

### 10.5 Revisions are authorized, but must remain intelligible

For every change, record the observed problem, proposed cause, exact change, parent checkpoint, data already inspected, added computation, and outcome.

- Numerical/integration repairs: preserve failed artifacts and replay only affected checks.
- LR/loss/gradientscope changes: record a training-policy revision and keep its trajectory visible.
- Structural or forward-equation changes: new candidate/version and explicit initialization; do not silently rename the old curve.
- Major changes after epoch 500 normally belong to the third candidate or a separately screened replacement, not an unlabelled mutation of the two finalists.

A modified candidate may be useful, but its mixed training history is not a matched pure-architecture comparison. State that rather than hiding the revision.

---

## 11. Resource strategy and occupied GPUs

### 11.1 Forecast the authorized workload instead of inheriting a diagnostic timeout

The default portfolio requires approximately **4,500–5,000 candidate-epochs**, including the two fresh controls and 2–3 H finalists. Two extra screens add at most 200 epochs; revised500-stage candidates must be counted explicitly.

Measure actual per-epoch time for each strategy after setup/warm-up, and forecast:

`remaining GPU-hours = sum_j(remaining epochs_j * median measured epoch seconds_j)/3600 + measured evaluation/setup allowance`.

For illustration only, 5,000 candidate-epochs at 20 seconds/epoch are 27.8 training GPU-hours; at 40 seconds they are 55.6. With two continuously available devices the training critical path is shorter, but startup, unequal workloads and contention can extend elapsed time. These are arithmetic scenarios, not hardware measurements or promises.

After the first 5–10 genuine epochs per profile, write a short ETA/resource note. Do not silently shorten 1,000 epochs to 1,000 updates to meet a previous six-hour budget. Also do not double the strategy menu because both GPUs happen to be idle.

Training is the main resource use. Keep ordinary evaluation/reporting near 15–20% of the campaign compute by sampling early and streaming full fields only at milestone reviews. Reuse saved predictions for plots; avoid repeated model calls for cosmetic re-rendering.

### 11.2 GPU assignment

Use one independent training job per authorized physical GPU where possible, with the next queued candidate ready. Verify physical GPU 1/2 identities at launch. Do not assume a remapped logical `cuda:0` is physical GPU0. No unauthorized GPU is used.

For each job, record batch size, microbatching, precision, peak memory, epoch time and a coarse contention status. Report isolated and contention-affected timing distributions separately. Prediction-quality comparisons may use contended runs if the math/data are unchanged; speed rankings must not use them as clean benchmarks.

### 11.3 If an external process occupies a device

Do not kill it, wait indefinitely, or abandon the scientific campaign merely because it is present.

1. Use the other authorized GPU and run the queue sequentially there.
2. If safe memory remains, use a smaller microbatch with accumulation on the contended device; retain the effective batch and full data pass.
3. Reduce receiver chunking/enable the existing activation checkpointing before changing model width, data, or physics.
4. At an ordinary checkpoint boundary, migrate an owned job to the other authorized GPU if beneficial. Preserve model, optimizer, scaler, RNG and data position; disclose the migration and any numerical reproducibility limit.
5. Use temporarily unavailable time for CPU data preparation, shared-code tests, saved-output evaluation, visualization and manual 5k packaging.
6. If neither device can execute even the smallest valid batch and the obstruction persists, report the concrete blocked operation and leave a tested resumable queue. An outstanding campaign is **not** `Goal complete`. Do not spin in repeated long sleep/poll loops.

Do not reject epoch evidence just because an unrelated process inflated its runtime. Mark the runtime as contention affected, as the user requested.

### 11.4 Progress and interruption

Maintain one human-readable campaign status page: run, strategy, complete epoch, latest key metrics, current action, and ETA range. Update at least every 100 epochs or roughly every two hours of an active job. Persist ordinary checkpoints every 25 epochs and at 100/500/1000, plus best/latest according to existing conventions.

Use small resumable stage jobs with no artificial idle waits between approved stages. The candidate ladders authorize continuation through 1,000 after the stated scientific selection. A user stop overrides that authorization: save owned-job progress and report completed versus outstanding work without launching replacements afterward.

No new monitoring daemon, cryptographic inventory, or contract-freeze framework is needed. Use the existing run store, logs, process management and security primitives.

---

## 12. Forward and hypergraph evaluation

### 12.1 Fair accuracy comparisons

Use the same physical input, query coordinates, masks and canonical target transformations for all compared models. Record full-normalized multichannel error for continuity with past work, but do not let it replace physical per-channel RMSE/MAE.

Main outputs: fluid `[u,v,p,omega,T]`; near-interface and far-fluid regions; surface temperature; `q_normal` proxy; material temperature; initial/final h/T ports; per-module material peaks; maintained inlet/outlet pressure difference.

Report equal-case means, pooled errors, p90/max tails, and M/context strata separately. Do not mix units in a physical scalar score. Train-derived dimensionless scores can support selection if their definition is explicit.

At 100 use a fixed stratified development panel plus cheap sampled validation. At 500 and 1,000 perform full native-grid evaluation of the canonical 89/90 development population for the surviving candidates/controls. Use streamed chunks. Do not repeatedly decode all fields separately for each plotting script.

Retained Run1804 is a mature contextual baseline; the fresh B-native/B-fine curves provide same-age controls. Fresh-versus-mature comparison does not isolate architecture, and all these legacy cohorts remain development evidence.

### 12.2 Graph quality

For each finalist, report by phase/mechanism:

- admitted and nonredundant K distributions, including source-empty/rescue/fallback cases;
- query participation, source-set overlap, group mass and distinctness;
- actual unique fine pairs, repeated paths removed, executed rows and padding;
- same-M/different-layout and same-layout/different-supported-context comparisons;
- capacity saturation and the difference between allocated and actually used groups.

Required interventions at identical physical weights:

1. full access, normal controls;
2. root union with its extra work disclosed;
3. source/degree/weight-matched geometry control;
4. genuinely effective rewire, with changed native pairs verified;
5. group controls set to identity while preserving access;
6. source groups exchanged while preserving work where possible;
7. a training-population fixed organizer summary;
8. fixed-K/frontier variant as an intervention, explicitly distinguished from independently retraining a fixed-K model.

A local gain in one case is a witness, not a population conclusion. Positive all-role wins and adverse role changes should both appear. A graph with more than two members permits collective dependence; it does not demonstrate a unique physical many-body decomposition.

### 12.3 Invariance and boundary behavior

Check source/module permutation, padding, query order and chunking, and the correctly defined quadrature-atom splitting test. Do not require an entire learned geometry tree to remain identical when its receiver indexing data are changed outside the tested equivalence.

Evaluate smooth physical-input paths with fixed and recomputed topology separately. Find actual trained topology changes before claiming a switch-continuity result. Hard group/source boundaries can create discontinuities even when within-plan gradients agree. For inverse use, freeze discrete connectivity over a local proposal step and re-evaluate after rebuilding; keep continuous geometry and physics live.

Record exactly which coarse/global/local channels bypass organization. H candidates must not reconstruct the same interactions through an undeclared Dense field branch.

### 12.4 Training and inference efficiency

Training: seconds per complete epoch, cases/second, queries/second, actual optimizer steps, extra response/shadow work, peak memory, and time to a declared quality threshold. Report medians and tails with contention separated.

Inference: synchronized repeated complete-wrapper calls on one low-M and one high-M case at small inverse Q and full field Q. Include encode, plan, P0/P1/P2 preparation and decode. Also time prepared decode as a separate scope. Do not add overlapping scopes.

Default to the correct dense-masked reference during initial development. Use an exact rectangular subset/block path only after output and first-gradient comparison. The existing packed path does not support every current direct-MM plan; do not route an unsupported plan through it or label unmeasured packed work as saved. Implement one generic supported execution path for the finalists, not three experimental kernel systems.

If there is no latency saving, report that. A faithful, reusable graph can still be a worthwhile finalist; an order-of-magnitude avoidable policy overhead remains a concrete engineering problem to expose.

---

## 13. Inverse-use evaluation: complete a meaningful task, not a large weak sampler matrix

Freeze each finalist before its inverse evaluation. No inverse loss may update its organizer/forward weights in the frozen-reuse comparison.

### 13.1 All finalists: finite-response and local heat-inference tests

First evaluate existing, independently recorded finite responses and pressure/peak increments. Compare against Run1804 and the zero-change predictor. Report absolute errors where response denominators/floors are unresolved.

For a manageable inverse test, use **known geometry and fixed-total nonnegative heat allocation**. Keep observations at fixed physically named sensors and retain disjoint held sensors. Visible geometry, material/context and supplied total heat are allowed inputs; hidden individual heat values are evaluation targets only.

Run the same small multi-start observation-matching procedure with every finalist and Run1804, for example, 12 development cases, 3 starts, and 30 steps with a common optimization budget. Calculate all trial fields with the same selected forward version. The heat variables must remain caller-owned differentiable tensors; a NumPy/provider conversion must not silently cut their gradients.

Before interpreting recovery error, measure the observation Jacobian rank/conditioning or selected singular values on the fixed-total subspace. Nonunique heat recovery must not be called a wrong physical design solely because it differs from one stored allocation. Conversely, a lower surrogate residual is not independent physical validation.

Compare observed and held-sensor residuals, heat/geometry feasibility, reproducibility across starts, per-module peaks and pressure behavior, evaluation cost, and whether graph-guided coupled updates improve on size-matched ungrouped updates. Optimization trails must be labelled as such.

### 13.2 Best qualified organizer: a small frozen generative comparison

For the best structurally meaningful finalist, train two equally initialized conditional inverse heads: graph links versus full links, using the **same frozen forward embeddings**, observation encoder, tasks, noise stream, optimizer and update budget. Keep the repaired nonlinear spatial observation conditioner.

Use the stable velocity-target formulation. For heat, avoid unconstrained logit magnitudes that merely saturate a softmax. A documented option is a zero-sum normalized fraction state:

`r0 = sqrt(M) * (heat/total - 1/M)` on active modules,

with centered Gaussian noise, inverse affine reconstruction, and an explicitly recorded simplex projection for the candidate proxy/final feasibility. Projected and raw quantities remain distinct; structural validity is not evidence of conditional quality. A pre-existing equally stable heat parameterization may be used if its semantics and controls are tested.

The current `BoundedVelocityPacketDiffusion` validates generated clean coordinates in `[-1,1]` and clips its provider proxy to that position range. The heat state above can exceed that range when M is larger than two. Therefore implement a separately identified centered-simplex heat adapter/model, or an orthonormal simplex-subspace parameterization with its own explicitly verified support. Do not feed this heat state into the position validator, clip away legitimate heat concentrations, or weaken historical position checks. Preserve zero-sum noise and the supplied total; handle M=1 as a deterministic, zero-free-dimension case. The same velocity algebra can be reused, but the state/proxy/decoder contracts are different. [S10]

Authorize up to 1,500 updates per inverse arm, with a training-only conditioning/overfit check and reviews at 200/750/final. Use a small complete evaluation, e.g. 12 tasks × 4 independent noise draws per arm, rather than hundreds of nearly identical controls. Save each completed draw immediately, including actual time indices, observations and held predictions.

Required comparisons: original versus valid alternative observation tasks from stored evidence where available; observation removal/shuffle as a labelled model intervention; same-weight effective graph/full intervention; sample diversity conditional on validity. Perturbing one sensor to an unsupported combination is not a new physical target.

If no finalist has a meaningful active graph, still complete the finite-response/local inverse-use evaluation for all finalists. Run the generative comparison only as an explicitly labelled active forced-plan diagnostic or report why the graph-reuse question remains unavailable. Do not present two full-access copies as a successful test of hypergraph benefit.

### 13.3 Independent physical checks and solve authorization

Use stored references by default. New Wind CFD is not part of this campaign. For Thermal, use only any **already verified remaining local-solver allowance**; the new epoch authorization does not erase the standing solve limit. A useful optional final check is up to 8 preselected nominal-grid baseline/candidate solves, balanced across methods/tasks, only if the allowance actually covers them. Otherwise report surrogate-only candidate outcomes and prepare, but do not execute, a separate validation request.

The Thermal analytic-wake/shared-grid generator validates its benchmark, not Navier–Stokes CFD, continuum conservation, or application-level safety. `q_normal` is a proxy. Convergence of the generator is not a geometry-discretization error bound.

---

## 14. Coding route and tests

### 14.1 Reuse the native workflow

Extend `Case_ThermalChannel/src/channelthermal/workflows/train_forward.py` and the shared registry/configuration for the campaign profiles. Keep ordinary `train.py`/`evaluate.py` entrypoints and native bucketed full-data epochs. Do not make a new giant one-off trainer that reproduces the entire physical wrapper.

Proposed shared modules, names adjustable after local inspection:

- `interface_fields/typed_hypergraph_state.py`: typed, target-free plan/state and diagnostics;
- `interface_fields/adaptive_receiver_hypergraph.py`: H-tree;
- `interface_fields/overlap_control_hypergraph.py`: H-overlap and H-local flag;
- a shared source-moment and unique-pair read helper;
- central registered capability metadata for dimensionality and inverse export.

Prefer reusing existing `MechanismPlan`, source-measure, sparsemax/gate, prepared-case and reader utilities. Add only the small missing abstractions; do not rename/rewrite every historical backend.

### 14.2 Common inverse export

Provide one shared export/evaluate interface for a selected candidate containing:

- physical source IDs/types/coordinates and measures;
- group admission and source membership;
- phase and frozen/recomputed topology semantics;
- receiver access callable or plan;
- low-dimensional group controls and provenance of global inputs;
- meaningful work counters.

The Thermal wrapper currently excludes interface-field architectures from its legacy `extract_hypergraph_plan`; do not route new models through that incompatible legacy exporter. Add a separate common typed export and have both wrappers call it.

### 14.3 Focused required tests

- All-access/control-identity versus B-fine, outputs and first gradients.
- Fine-source preservation and unique-pair deduplication.
- One environmental normalization over the source union.
- Correct source measures/native amplitude factors under changed K.
- Parent-to-identical-child identity and a real trained/nonidentical split effect.
- Source/receiver permutation and physical-ID tracking.
- Zero/one/many active modules, padding, empty group/source type, and finite masked reductions.
- Masked pooled maxima/means for selector or group descriptors; explicit empty flags, no fake zero extremum dominating a negative-valued valid population.
- Nonzero useful organizer gradients and no physical-gradient leakage from the shadow.
- Actual optimizer coverage, zeroing, resume state and curriculum continuity.
- Query batching/phase freshness and actual executed-row counting.
- Input-only inference: no target/reference outputs, hidden inverse targets or case-ID embeddings.
- Native Thermal predicted-port one-step execution and disposable 3-D Wind compatibility.

Tests support software claims only. They do not replace the requested 100/500/1,000-epoch runs or reference evaluation.

---

## 15. Final report and visualizations

Open with a plain-language table for **Predictor / Organizer / Inverse**: what changed, what improved, what failed, evidence strength, and the one resulting next decision. Then include a short strategy leaderboard and a diagram of the aligned shared core/case wrappers.

Required main figures, using saved numerical evidence:

1. **Training ladder and maturity:** real epochs, case passes, learning-rate/structure schedule and GPU-hours; control and candidate validation curves; mark modifications and contention.
2. **Physical fields:** representative and difficult full fluid fields and residuals with a common scale, plus Thermal interface/material curves or maps. Include pressure and peak errors, not temperature-only successes.
3. **Actual hypergraphs:** source memberships and receiver supports on physical coordinates; K distributions by case/phase/M/context; distinguish near envelope, active groups, fine sources, and full bypasses.
4. **Group utility:** same-weight normal/root/geometry/rewired/control-identity comparisons at stated work. Show failures and full fallbacks.
5. **Fidelity and efficiency:** error/work and time-to-quality; actual rows versus allocated/selected pairs; isolated/contended timing labelled.
6. **Response and inverse use:** finite field/material responses, observed/held sensor errors, generated or optimized heat allocations, and a small number of complete design/sample trails with exact time/iteration axes.
7. **Manual 5k decision:** which 1–2 candidates should be matured, why, and what unresolved risk the run will test.

Use existing plot utilities and save raw arrays once. Inspect selected figures before delivery. Masks, units, sampling/quadrature, clipping percentage, reference source, checkpoint and exposed-development status must be stated. Latent group indices must not be matched across models as physical identities without a physical-support-based matching procedure.

Keep detailed quantities in appendices. The first two pages should be readable without reconstructing all past reports. Do not equate a successful software suite with scientific success.

---

## 16. Manual 5,000-epoch handoff

For each of 1–2 selected finalists, prepare:

1. exact selected and e1000 checkpoint locations, core/case configs, local-surrogate binding and normalization;
2. a continuation config to train from the **e1000** state to **total epoch 5000**, preserving optimizer/scaler/RNG and the established objective;
3. a separately labelled fresh-seed 5000 config if the user wants an independent long run;
4. a tested launcher accepting physical GPU 1 or 2, with a clear run name and output location;
5. command examples with all actual required arguments filled in;
6. approximate duration based on measured epoch time, plus memory and contention guidance;
7. evaluation/plot milestones and an ordinary interrupt/resume procedure.

Verify parser/config resolution, checkpoint loading and a small native forward for the continuation path. Do **not** instantiate a 5000-epoch optimization process during verification. Avoid copying a best-e940 checkpoint and claiming it resumes at 1001; the exact source epoch determines continuation.

Where the trainer resumes in the checkpoint's parent directory, create a properly identified continuation workspace without altering the preserved finished 1,000-epoch result. Use existing run-store primitives and supported arguments rather than silently rewriting historical artifacts.

Recommend on a combined fidelity/organization/response assessment, not the lowest scalar training loss. If no candidate establishes the intended graph benefit, still provide executable recipes for the most informative exploratory 1–2 candidates and label them **exploratory, not validated HONF deployment**. Do not invent a winner to satisfy the handoff.

The user alone launches the 5,000-epoch jobs.

---

## 17. Completion and repository practice

Preserve existing security and artifact rules. Reuse trusted checkpoint loaders and native context validation. Do not delete safeguards, disable hooks, or introduce new cryptographic hashes/contract freezes/baseline snapshot systems as default infrastructure.

Scientific screens classify evidence and allocate the already authorized run budget. They are not new blocking approval gates around ordinary code execution or heuristic validation. Never substitute a mocked pass for a real optimizer step, model replay or reference result.

Commit durable shared source, case configuration/workflow changes, reusable tests, the final report and the manual launch recipes. Keep checkpoints, raw data, generated figures and one-time renderers in the prescribed ignored local paths. Inspect the entire outgoing history, run the existing pre-push hook, push the current non-default branch, and verify local/remote tip equality.

**Do not report Goal complete until the 2–3 final candidate runs really reach 1,000 epochs and the requested comparison/handoff is delivered, or the report explicitly states which requested deliverables remain incomplete because of an unresolved external limitation.** Finishing fast or consuming all available hours is neither success criterion.

---

## Source and method notes

[S1] `HONF_Proj/docs/reports/_bk/20261002_233129Z_HONF_Directed_Packets_and_Stable_Inverse_Implementation_Report.md`, scientific implementation 0076b28. Use the updated relative-link version if present. Main evidence: Predictor; Availability/ranking/calibration; Group utility; Thermal predictor; Inverse; Resource accounting.

[S2] `HONF_Proj/Case_WindFarm/src/windfarm/model.py`, 0076b28. Shared-core wrapper, allowed architecture list and historical 3-D sparse-incidence configuration workaround.

[S3] `HONF_Proj/Case_ThermalChannel/src/channelthermal/model.py` and `interface_field_coupling.py`, 0076b28. Thermal physical wrapper, local ports, shared-core calls and legacy-only plan exporter.

[S4] `HONF_Proj/src/honf_forward_core/interface_fields/core.py`, 0076b28, `InterfaceFieldCore.__init__`. Architecture-specific context/backend construction.

[S5] `HONF_Proj/src/honf_forward_core/interface_fields/three_term_context.py`, 0076b28. Background plus two fine terms; source-free coarse placeholder; zero local-context branch.

[S6] `HONF_Proj/src/config_core/forward/dense_pairwise_interface_context.json`, 0076b28. Current native baseline profile. Compare against actual archived Run1804 resolved configuration before claiming reproduction.

[S7] `HONF_Proj/Case_ThermalChannel/configs/case_default.json` and `src/channelthermal/workflows/train_forward.py`, 0076b28. Full-case dataset/loader, predicted ports, local model, native losses, initialization/resume and real epoch loop.

[S8] Repository `AGENTS.md`, 0076b28. Durable-only upload, whole outgoing-history audit, report gain/miss narrative and inspected measured figures.

[S9] `HONF_Proj/src/honf_forward_core/interface_fields/directional_packets.py`, 0076b28, `score_cases` and `plans_from_scores`. MM-only residual source scores, zero split logits, external frontier cuts, and default-root evaluation.

[S10] `HONF_Proj/src/honf_inverse_core/models/bounded_velocity.py`, 0076b28. Velocity loss and sampler, normalized-position target bounds and clipped provider proxy. The proposed heat-simplex adapter is new work, not already supported by those position contracts.

[M1] Chien, Pan, Peng and Milenkovic, *You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks*, ICLR 2022, arXiv:2106.13264. Inspiration for learned multiset source/group/receiver operations, not HONF physical validation.

[M2] Martins and Astudillo, *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*, ICML 2016, PMLR 48:1614–1623, arXiv:1602.02068. Sparse normalization and its differentiability properties; no implication that sparse attention weights identify causal physical sources.

[M3] Louizos, Welling and Kingma, *Learning Sparse Neural Networks through L0 Regularization*, ICLR 2018, arXiv:1712.01312. Stochastic admission/gate machinery; no guarantee of conditional K, fidelity preservation, or GPU speedup.

The candidate equations, shared-core campaign, training schedules and selection rules above are proposed research designs. They are not results reported in those sources.
