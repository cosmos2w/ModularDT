# HONF 1503-v5: task-trained functional coalescence

## Decision, scope, and evidence

**Do not extend Run 1506 unchanged. Keep Run 1502 as the scientific accuracy/implementation base. Retain the continuous-function transform and exact source-moment quotient from v4, but replace passive discrepancy-triggered contraction with a small task-trained, case/phase-conditioned detail controller.**

The central research question is:

> Can the model learn which distinctions between interaction functions are unnecessary for reconstructing the physical outputs, eliminate those distinctions continuously, and pack equal functions without discarding the physical information they organize?

This plan follows source commit `427b24f253cdfd2717a45d52851daa39e25794db` on `agent/honf-core-next`. The new experiment is conceptually **1503-v5**; use the existing allocator for its numeric run ID. Do not assume that an ID is free, overwrite Run 1506, or describe changed mathematics as a continuation of it.

This plan distinguishes three kinds of statements:

* **Observed:** facts in `HONF_Run1506_Continuous_Functional_Coalescence_Diagnostics.md` and source at `427b24f`.
* **Derived:** algebraic consequences of the specified operators.
* **Proposed:** new modelling choices that need actual training and physical-output evaluation.

The old report and its selected checkpoints remain valid historical evidence. No historical checkpoint is edited. All 90 `test` IDs are development data because the same split supplies sampled validation. Do not call them an untouched test population.

### What the previous experiment established

At exact e500 and all three distinct selected-weight checkpoints, the surveyed v4 models retained R=12 and had zero transition and closed nodes. The minimum node RMS scores were 0.188, 0.199, 0.230, and 0.208, respectively, all above the 0.06 keep threshold. Compact and virtual query statistics therefore coincide. The exact endpoint's equal-case mean fluid L2 is 0.12839 versus 0.10568 for parent e500. Its selected e459 checkpoint is substantially better (0.09870), but that is not an exact-e500 comparison or evidence that any merger helped.

The all-90 matched-chunk application ratio is 1.990, even though no functions closed. Fine physical rows remain rectangular. The runtime cost includes probes, all-node hypothetical-tie scoring, and a combined diagnostic on every active preparation.

The training-derived tree already includes all three recurring pair types observed in 1505: [0,10], [1,8], and [6,7]. Tree restriction may limit other opportunities, but absence of these three pairs is not the immediate reason for v4's failure.

### What is not established

The full 500-epoch training stream was not surveyed for every transient contraction. The endpoint/selected-checkpoint surveys do not prove that no training mini-batch ever entered a transition. The observed trajectory difference after e50 is not explained by a single CPU same-weight forward check. The routing images are not evidence of physically causal groups. No native closure was observed in the bounded geometry sweep, so trained closure continuity was not verified.

## 1. Root-cause diagnosis to preserve in the final report

### 1.1 Passive compression had no direct learning signal in its operating region

In `organization/functional_fusion_tree.py::node_contraction`, sufficiently large squared scores map to exact identity retention s=1 and contraction gamma=0. In that region ds/d(score)=0. The existing loss has no separate objective encouraging fewer access functions. Consequently the contraction branch supplies no direct gradient to approach a merge there. Reconstruction can incidentally change the score, but there is no reason it should move into a narrow prescribed band.

The train-derived pair minimum was already 0.203206, above 0.06, and the median was 0.728923. This was a criterion-scale mismatch before the formal run, not a problem that another 450 unchanged epochs was likely to diagnose efficiently. A 0.02/0.06 score cutoff is not a 2%/6% physical-field error guarantee.

**Revision:** learn the residual-function retention from actual task loss, with an explicit differentiable structural-complexity objective. The new objective is justified by the observed absence of a contraction gradient and any successful closure. Do not merely increase the old cutoff until the R histogram becomes attractive.

### 1.2 The score is a worst-probe/mean-energy ratio

`_score_route_changes` computes, for each probe role r and source-action block b,

    numerator = max_q (delta_alpha_q^T G_b delta_alpha_q)
    denominator = sum_q w_q (alpha_q^T G_b alpha_q)
    score_squared = max_(r,b) numerator / (denominator + 1e-8)

It is not the ordinary weighted RMS of the route-induced action difference. It mixes a maximum numerator and a mean denominator, then takes another maximum across roles and blocks. A small parent action block may dominate the ratio without dominating physical output error. The latent-action metric is useful for diagnostics, but is not an output-preservation certificate.

**Revision:** move this calculation out of routine inference. Use real target losses as the optimization signal. On a bounded training-only panel, compare old scores with actual tied-versus-untied physical errors to understand false rejections; do not silently change historical score semantics.

### 1.3 Identity mathematics did not use the parent computational path after e50

The active v4 `prepare` unconditionally builds scores, transform, quotient incidence/moments and `combined_post_transform_discrepancy`; it sets `coalescence_all_singleton=False`. Its `_route` obtains parent logits, applies another contraction and calls weighted sparsemax even with multiplicities one. This is a different finite-precision operation/gradient sequence from direct parent execution.

This explains additional work and provides a plausible numerical source of training divergence, but does not prove that it caused all endpoint error. Investigate actual first divergence under identical batches/RNG/optimizer state. Do not label an inert merger harmful to physics on the basis of separately evolved weights.

### 1.4 Fewer class labels cannot by itself remove dense fine physical rows

Even successful coalescence changes the group/control axis, not automatically the expensive QM/QE source-pair axis. The new experiment will not claim physical row sparsity or a large speedup from R alone. It must at least avoid the current preparation tax. A separate exact-reader project can follow when formation is useful.

## 2. What to retain and what to remove from the new hot path

Retain:

- Run-1502 fine MM/ME/EM contextualization and physical coupling.
- K=12 provisional functions, D=16 controls, source normalizers, query sparsemax.
- The existing eleven-node train-derived binary tree for the first candidate.
- Continuous transformation of the original finite access logits.
- Multiplicity-correct query density/mass and conserved source-resolved moments.
- Original fine source states, coordinates, physical measures, environmental K/V and value modulation.
- Provisional current-phase kappa and the existing finalization convention.
- One physical environmental softmax over the complete supported source union.
- Rectangular training/reference execution, phase-local preparation, query-chunk reuse.

Remove from the **new** routine prediction path:

- Probe catalogues used solely to decide contraction.
- Per-node hypothetical ties and four-block Gram scoring.
- Combined post-transform diagnostics when not explicitly requested.
- Online ADMM, iterative cluster searches, and CPU per-case clustering.
- Dependence of a contraction decision on the requested query grid.

Historical implementations remain loadable and readable. This is not permission to change their scientific operators. Exact execution-only improvements to old paths, if made, must be separately identified and tested.

## 3. Mathematical core: remove function differences, not source information

Let the existing phase-local access logits be l^0(q) in R^12. Define

    l_tilde(q) = T(c, p) l^0(q),

where c denotes physical case inputs/current states and p denotes the physical phase. T is shared by every query in that prepared phase.

For a tree node C, P_C averages the entries belonging to its subtree and leaves other entries unchanged. A detail-retention coefficient s_C in [0,1] gives

    M_C = P_C + s_C (I - P_C),
    T   = product_C M_C.

For s_C=1, that node does not alter the functions. For s_C=0, every descendant function becomes equal after the subtree's transformation; compaction is then an exact representation change. Intermediate values continuously attenuate differences. Do not substitute new centroid-distance functions or renormalized averaged keys when packing.

This changes the set of functions the model implements during training; exact quotient equality is to the corresponding transformed virtual model, not to unmodified parent predictions.

### 3.1 Fast equivalent tree representation

The tree is laminar. Its averaging projections commute. Construct an orthonormal Haar basis H over the proposal labels once from the fixed tree:

- h_0 = (1,...,1)/sqrt(K).
- For internal node C with children L,R of sizes n_L,n_R, h_C is

    +sqrt(n_R / (n_L (n_L+n_R))) on L,
    -sqrt(n_L / (n_R (n_L+n_R))) on R,
    0 elsewhere.

For each internal contrast n,

    d_n = product_(C ancestor of n, including n) s_C.

Then

    T = H diag(1, d_1, ..., d_11) H^T.

This is algebraically the same transform, not a new model. Use precomputed ancestor masks and batched products/matrix multiplies rather than eleven Python-driven row-update loops if it benchmarks better. H and ancestor masks are small ordinary model buffers. Do not add an eigensolver in the forward pass.

This basis is a basis of **proposal-function differences**, not a claim that its vectors are physical Fourier/PDE modes.

In exact arithmetic, for fully active leaves:

    R = rank(T) = 1 + sum_n 1[d_n > 0].

Use known zero-retention subtree metadata for packing. Do not detect classes with a fuzzy floating equality threshold, estimate rank numerically, or prune tiny nonzero d_n. Floating underflow is not a scientific merger.

### 3.2 Source-moment quotient

For an exactly closed class C_r of multiplicity m_r:

    Abar^S_(s,r) = sum_(k in C_r) A^S_(s,k),
    B^S_(s,r)    = sum_(k in C_r) A^S_(s,k) h_k,

where S is module or environment source type.

Use

    a_(q,r) = m_r [L_r(q) - tau_q]_+,   sum_r a_(q,r)=1,
    b_(q,r) = a_(q,r)/m_r,
    rho^S_(q,s) = sum_r b_(q,r) Abar^S_(s,r),
    n^S_(q,s)   = sum_r b_(q,r) B^S_(s,r).

L_r is a representative row of the **same transformed original function** T l^0. a is compact probability mass; b is per-constituent density. Only b enters the source contractions.

The environment value modulation still uses sum_r B^E_(j,r)=sum_k A^E_(j,k)h_k. Project score moments into head space without forming Q x E x D banks. Keep the current provisional kappa; do not replace K=12 by R inside the parent's amplitude calibration.

The field remains C_g+C_M+C_E. No extra field branch, averaged physical-source value bank, or per-group attention is introduced.

## 4. The genuinely new component: a learned detail controller

A small shared network predicts one finite scalar a_C(c,p) for each of the eleven fixed nodes. It uses only information already available in provisional preparation.

Recommended implementation:

- For each child subtree, compute unweighted valid-leaf mean/variance summaries of existing normalized query keys and group controls.
- Include module/environment masses, normalized source-type centres, occupancy, child sizes, and their differences; include current global control.
- Use symmetric combinations (sum/mean and squared differences) when child order is arbitrary.
- Use normalized physical coordinates and explicit zero-mass conventions. Avoid dividing by tiny learned mass without a valid branch.
- One shared two-layer MLP of hidden width 32 or 64 outputs the scalar logit for every node in one batched call.
- No case-ID embeddings, labels, target fields, hand-coded expected R, or fixed module-count-to-R mapping.
- The same controller architecture and weights are used in P0/P1/P2; their physical states differ.

The fixed tree preserves an existing structural prior; the controller makes its detail retention input dependent. A lower R in every case is not automatically useful adaptivity. Assess whether replacing case-dependent controller inputs with their train-population mean changes decisions and task errors.

Initialize the new controller without changing the parent's parameter initialization/data RNG stream. Use ordinary RNG isolation during construction. An initial logit around 1.6 is the starting choice below, not an established optimum.

## 5. Primary learning method: reversible stochastic detail retention

Use the hard-concrete construction as a **training relaxation of residual-function retention**. It does not permanently delete prototypes or any physical source.

For node logit a, independent node noise u in (0,1), beta=2/3, lower=-0.1, upper=1.1:

    v = sigmoid((log(u) - log(1-u) + a)/beta),
    z = clamp(lower + (upper-lower) v, 0, 1),
    s = z^2 (3-2z).

The final cubic is a proposed smooth endpoint transform. It leaves exact zeros/ones unchanged and makes the retention derivative zero at those endpoints. It is not part of the original hard-concrete paper. Its positivity event remains exactly z>0, so the analytic probability below remains valid.

At deterministic evaluation use the explicit proxy

    z_hat = clamp(lower + (upper-lower) sigmoid(a), 0, 1),
    s_hat = z_hat^2 (3-2z_hat).

This deterministic proxy is **not** the stochastic mean or median. Never silently average stochastic fields at evaluation or replace it with thresholded binary decisions. Train the deterministic path as well, as specified below, and inspect any stochastic/deterministic gap.

The probability that the base retention is positive is

    p_C = sigmoid(a_C - beta log(-lower/upper)).

All parameters and original proposal functions remain registered after closure. For finite logits, stochastic training can explore reopened detail. This avoids irreversible structural deletion; it does not guarantee that an extremely saturated controller will reopen easily. Track saturation and use the bounded remedies below rather than claiming guaranteed reversibility.

### 5.1 Expected compact count, not an arbitrary gate sum

Conditional on the current phase descriptors, sample node noises independently. For all twelve proposals occupied,

    E[R | c,p] = 1 + sum_n product_(C ancestor of n, including n) p_C.

The ancestry product accounts for a closed ancestor hiding descendant distinctions. Simply summing eleven keep probabilities would count detail that a closed ancestor already removes.

For partially occupied proposals, disable closure at a node containing inactive leaves and preserve the existing continuous mass-eligibility fade. When eligibility is below one, that node cannot close exactly and its positive-retention probability is one. Compute expected class count by a bottom-up recursion with inactive leaves returning zero:

    E[R_C] = p_eff,C (E[R_L]+E[R_R])
             + (1-p_eff,C) * 1[any active leaf in C].

Here p_eff=1 when closure is disabled, otherwise p_eff=p_C. Validate this formula against actual sampled partitions for masked cases; do not use the fully occupied formula blindly.

### 5.2 Accuracy objective and structural pressure

Preserve the existing physical task loss and its masks/normalization. Add a clearly separate expected-complexity term:

    J = 0.5 L_task(deterministic controller)
        + 0.5 E_u L_task(stochastic controller)
        + lambda * C,

    C = mean_(case,phase) (E[R] - 1)/(K_occupied - 1),

with C=0 for K_occupied<=1.

The role of lambda is explicit: maintaining an extra interaction distinction must earn its cost through better physical prediction. It is not a required target R, an output-approval threshold, or a penalty on source magnitude.

For efficiency, estimate the two task terms by assigning approximately half the cases in each training batch to deterministic control and half to stochastic control. Keep a case's assignment the same throughout P0/P1/P2. This gives one physical forward/backward per case rather than two. For very small batches alternate assignments over steps. Record the estimator. Evaluation is fully deterministic.

Compute C using controller inputs detached from the physical source network, while keeping the controller parameters live. This means structural pressure directly updates the detail controller, not source masses/features in a way that makes them artificially cheap. The physical network still learns through actual task loss and the live predictive controller. This is an intentional stop-gradient/block-training choice, not the full derivative of an unrestricted regularizer through all source features. A second tiny controller call on detached inputs is acceptable; another physical forward is not needed.

Do not put C only in detached diagnostic dictionaries. Connect it to the trainer as a live scalar, once per prepared physical phase, not once per query chunk. Count and average actual phases consistently.

### 5.3 Initial scale and bounded adjustment

Do not replace v4's arbitrary read-error threshold with an unexamined large lambda.

On real train batches during the short pilot, measure task and complexity gradients with respect to controller parameters. Choose an initial lambda making its median gradient norm approximately 10% of the task-gradient norm. Use only finite, nonzero observed gradients; if task gradients are absent, diagnose the actual path instead of dividing by epsilon and claiming calibration.

This is an initialization heuristic, not an accuracy certificate. The allowed pilot may try one factor-three stronger or weaker lambda based on observed non-response or fidelity loss. Do not tune on the 90 development routing figures.

Parent optimizer settings remain unchanged. A separate ordinary optimizer group for the new controller may start at 1e-3; the parent remains at its existing rate. This new-group learning rate and all changes selected in the pilot must be recorded. Do not silently change the whole optimizer.

## 6. Gradient, sampling, and boundary requirements

- Sample B x 11 controls once per prepared phase, never per query or source pair.
- Reuse those live controls through all inner/outer receiver chunks of that preparation.
- Put sampling outside activation-checkpoint recomputation regions, or explicitly preserve the relevant RNG state. Backward must not resample a different operator.
- Keep the deterministic/stochastic case assignment constant across physical phases of the same forward.
- No ADMM/unrolled numerical optimizer and no score-threshold straight-through estimator is needed.
- Finite-difference tests use deterministic evaluation; stochastic path tests use common fixed noise at both endpoints.
- The exact closure condition is the implemented retention plateau s=0, with topology obtained from that condition. Do not turn a small positive retention into zero for a prettier R distribution.
- Packing must reproduce the corresponding virtual function, including first derivatives of the controller in the continuous region.
- Do not infer physical-design smoothness from tests that inject closed gates. Locate a native deterministic closure in trained physical inputs and shrink its coordinate bracket.
- A change of R can accompany a derivative kink without a field jump. Report one-sided limits and gradient behavior separately.
- Existing sparsemax/occupancy/zero-mass boundaries also need coverage. Do not claim a global smoothness theorem for the complete physical wrapper from continuity of T alone.

## 7. Practical execution route

### 7.1 Training: avoid dynamic packing overhead

Use a virtual width-12 predictive path for training first:

1. Prepare original fine sources and the parent controls once.
2. Predict eleven retention logits from existing summaries.
3. Construct T through the small Haar/ancestor form.
4. Route with ordinary masked sparsemax over T l^0.
5. Use the original source memberships and source-control banks in the fine reader.

This is the transformed virtual model and avoids CPU scalar extraction/Rmax changes during training. It also isolates the new mathematical effect from unnecessary changes in weighted sparsemax/source-bank arithmetic.

At inference implement the exact compact quotient and benchmark both compact and virtual execution. Both are the same constructed model within floating tolerance. Keep the faster execution policy at a given supported shape; report actual width and work. If virtual execution is faster and remains at width12, do not describe logical R as executed compact computation.

### 7.2 Hot-path exclusions and precision

The new forward must not call `node_source_action_scores`, `combined_post_transform_discrepancy`, or a functional probe builder to choose retention. They remain opt-in diagnosis/research tools for old/new models.

Use the installed PyTorch/CUDA environment, initially the parent's FP32 physical arithmetic and AMP policy. The algebra-check package was run on a separate CPU PyTorch version; it is not evidence about the production environment. Do not upgrade libraries globally, enable TF32/AMP silently, or add mandatory Triton dependencies.

Factor the existing finite query-logit calculation into a shared helper if needed so the new route does not compute and discard parent sparsemax before computing transformed sparsemax. Preserve the historical parent operation order and regression behavior.

Use fixed-shape batched operations for the controller and T. Optional `torch.compile` is an implementation attempt, not a prerequisite; compare compile startup, recompilation, and steady-state time. Fall back to vectorized eager if compilation is unhelpful.

Haar reconstruction can have ordinary FP32 row-equality roundoff. Exact class IDs still come from zero-detail nodes. For compact logits, compute one representative transformed function per class and repeat it only for parity references. Do not use numerical row clustering. Assess tolerances on the actual neural dtype and include parent repeatability.

### 7.3 What this can and cannot accelerate

Expected R measures access-function complexity, not FLOPs of the physical fine reader. Original nonlinear geometry access terms may still require all twelve proposal components; replacing a sum of distances by distance to an averaged centre is not allowed.

This round should remove v4's probe/Gram planning overhead and keep preparation near the parent while producing useful learned coalescence. It does not promise latency below Dense or 1404. No new selected-source kernel is authorized merely because the gate controller works. The prior exact fused reader remains a separate optional comparison, not the training default.

## 8. First execute a bounded causal/debugging review of Run 1506

Use actual static checkpoints and real batches, not synthetic scores alone.

### A. Explain the inactive score range

On a small stratified **training** panel, export per-role/block numerator, denominator, worst-probe index, and full-node score. Separate environment, physical ports, and outside-temperature probes. Compare maximum and weighted-mean numerators as different diagnostics; do not relabel the old metric. On a few low-score nodes, execute actual full-tie counterfactuals and compare physical field/port/flux errors. This assesses score usefulness without changing the historical model.

### B. Locate training-path divergence

Start both parent and identity-active v4 in memory from the same existing e50 checkpoint, optimizer state, real batch order, and ordinary RNG states. Compare forward, all relevant gradients, clipping, updates, parameter buffers, and RNG consumption over at most 50 consecutive actual update steps. Also collect node activity on those training batches. Compare a direct parent path, a forced-identity transformed path, and normal v4 where useful.

If the first difference is only reduction-order roundoff, record that and its growth without declaring it a semantic bug. If it is a real extra gradient, state/RNG mutation, or different input, fix the concrete issue and add a focused regression. Do not spend days reproducing all 500 epochs. Existing data/checkpoints and ordinary tests suffice; create no baseline snapshot framework.

On the surveyed e459/e500 weights, also replay the direct parent architecture with the same shared neural tensors as a labelled diagnostic. Because the stored surveys have T=I, this can test whether useful selected weights survive removal of the inactive wrapper on those cases. It is not a proof of equivalence on unseen inputs, where the old score might enter transition. Never overwrite the original model or claim a global export equivalence from that bounded replay.

### C. Remove evidence-only work from timing

Use the profiler to separate source preparation, controller/score work, quotient packing, and decode. In the new path, leave expensive diagnostic collection off by default. Confirm the actual call counts rather than trusting a `maps_off` label.

This review should take no more than about 60-90 minutes of targeted debugging. If some causal question remains unresolved, document it and proceed with the isolated new predictive path; do not invent an explanation.

## 9. Bounded active experimentation: do not wait 500 epochs for a dead mechanism

Codex is explicitly authorized to solve ordinary obstacles rather than report the first failure. Use a limited hypothesis-driven pilot before selecting the formal recipe.

### Budget

- At most **three** short real-optimization pilot recipes.
- At most **200 optimizer updates per recipe**, 600 total, and roughly three GPU-hours total, whichever is reached first.
- Fixed small train-only panel/batches covering low/high M and differing operating inputs. No development routing figures for tuning.
- Pilots may initialize from static Run-1502 e50 weights to isolate controller learning quickly. These are labelled warm-started diagnostics, not new same-epoch baseline claims.
- A formal candidate is fresh from the original seed; do not reuse pilot-trained neural/controller weights. Reuse only the documented chosen recipe.

### Primary attempt

Train the conditional hard-concrete detail controller with real physical outputs and expected R pressure. Verify separately:

1. the controller receives nonzero task gradients on executed intermediate retentions;
2. expected R has nonzero gradients even while deterministic controls are clipped open;
3. learned controller logits actually move;
4. deterministic, not merely sampled, closures become possible;
5. field/port/flux quality is measured together.

A synthetic closure or fewer stochastic samples is not success. Equally, lack of R change in the first few steps is not by itself failure when logits and task responses are moving.

### Authorized countermeasures selected by the observed obstacle

**No structural movement:** verify the live loss connection, phase reduction, controller learning rate, and clipping first. Then try the single factor-three lambda change justified by gradient measurements. Do not change topology thresholds.

**High stochastic variance / deterministic mismatch:** try antithetic uniform draws u and 1-u on a bounded subset or increase the deterministic-case fraction to 75%. Measure whether variance and deterministic physical errors improve. This is a second estimator attempt, not a different fine operator. Record any extra forwards and its cost.

**Rapid global collapse or port degradation:** reduce structural pressure or prolong its ramp; retain task supervision on deterministic cases and check physical loss masks. Do not impose a desired R floor or permanently ban a group. A common R=1 solution may be a static simplification, but is not evidence for useful case-adaptive hypergraphs.

**Implementation/memory bottleneck:** try fixed-width virtual training, batched Haar construction, and eager versus compiled execution, keeping the mathematical operator identical. No online solver or custom kernel detour.

**Most influential summaries missing:** one targeted controller-input correction is allowed if a real counterfactual shows insensitivity to an already-available physical variable. Do not grow a second encoder/attention stack or add case-ID features.

Not every obstacle requires all alternatives. Keep a short local record of the hypothesis, change, executed result, and conclusion. Resolve ordinary errors locally; the final report should emphasize meaningful findings and genuinely unresolved blocks, not a catalogue of every transient exception.

For a hard local blocker, use at most two substantially different targeted remedies or about 45 minutes before documenting it and moving on. Do not loop indefinitely. Never conceal a failed scientific mechanism or claim a fix without actual execution.

### Pilot selection

Choose a recipe from real deterministic physical performance and evidence of an active learned detail mechanism, not from smallest R alone. If all pilots fail to create a useful learning pathway, do not launch a ceremonial 500-epoch run; report the hard failure with the bounded evidence. This is experimental selection, not a new software-approval system.

## 10. Formal run and comparisons

Launch at most one selected fresh candidate. Retain the original Run-1502 task settings and seed, the fixed v4 tree, and the same fine reader.

### Training stages

- e1-50: exact parent predictive path. New controller construction must not consume parent RNG or affect the optimizer trajectory. The head is inactive here.
- e51-150: smoothly introduce retention through `s_eff = 1 - w(e)*(1-s)`, with w=0 at 50 and w=1 at 150, and ramp the complexity weight.
- e151-500: fixed selected recipe; no silent topology threshold, tree, loss, or precision change.

During w<1, stochastic base closures do not yet give actual compact closures. Report actual R separately from expected full-strength R. Do not claim success from the latter.

At e50 validate parent tracking and the imminent controller path through real optimizer checks. By e100/e150 inspect deterministic closures and per-node learning. Do not repeat the previous practice of allowing an obviously disconnected or permanently flat mechanism to run to 500 unchanged.

### Epoch-150 review

Evaluate exact e150 and the distinct validation-selected checkpoints on all 90 Q8192 grids; gather Q1024 P0/P1/P2 formation separately. Compare with static parent checkpoints under the same selection/epoch policy when available.

Review actual deterministic R variation, controller sensitivity to case inputs, fidelity across physical outputs and M strata, preparation/decode/application cost, and native boundary evidence. If retentions remain all open but are changing under a credible objective, a limited continuation can be justified; if the mechanism is dead, use the already bounded attempt policy or stop rather than spending another 350 epochs passively.

### Epoch-500 stop

Stop no later than 500. Evaluate exact endpoint and each distinct saved selected model independently. Report both pooled and equal-case field metrics when available, with unambiguous labels; do not mix L2 with RMSE. Each reported model gets its own formation and timing.

Provide a scientific recommendation about a later 5000-epoch run, not an automatic continuation. A credible recommendation requires useful deterministic case-dependent coalescence, acceptable physical tradeoffs at one checkpoint, no finite native topology-switch jump beyond numerical repeatability, and removal of the large preparation tax. Do not use arbitrary R targets or a single accuracy ratio as a software blocking gate.

## 11. Required correctness and empirical tests

### Algebra and implementation

- Haar transform versus the current projection-product transform for random retentions, unequal child sizes, nested zero nodes, and masked proposals.
- Compact versus virtual routing, rho, B moments, nonlinear QM/QE, value modulation, and first gradients.
- All-open predictive path against the direct parent; independent testing after e50 rather than only scheduled warmup.
- Known zero/one and intermediate detail coefficients, source-empty type, inactive proposals, tiny nonzero mass, and mixed batches.
- Analytic expected R versus Monte Carlo actual partitions; test ancestry and masked recursion, not just individual keep probabilities.
- Live complexity gradient reaches controller parameters and intentionally does not reach source-feature tensors through the detached cost input.
- Changing output query set/order/chunking does not change a deterministic prepared coalescence plan.
- Repeated deterministic inference is independent of prior stochastic training forwards.
- Same stochastic gate realization in checkpoint-recomputed backward and no per-query resampling.
- CPU import/fallback and actual production GPU optimizer steps, without assuming a library upgrade.

### Real physical evaluation

Keep field, near/far, fluid channels, internal/surface temperature, final port temperature, effective h and flux separate. Show M=3/5/7/10 strata and difficult-case tails. Inspect both raw and final port predictions so changes in the physical feedback loop are not attributed solely to P2.

Do not assert conservation from Abar/B identities. They conserve latent incidence/control sums, not physical energy or fluid mass by theorem. Use existing physical residuals/targets where available.

### Native boundaries

Search a small train-only panel with deterministic learned closures first, then use a fixed development panel for evaluation. Vary feasible geometry and, where the data support them, operating inputs. At an actual deterministic R transition, shrink the bracket and compare one-sided field, port and flux limits. Fixed-topology AD/FD must be reported separately. No found transition means the native-boundary claim remains unverified.

If no CFD derivative reference exists, label the result surrogate self-consistency. Do not create synthetic CFD truth from model outputs.

## 12. Physical usefulness rather than appealing colored regions

The current figures show learned routing mass, exact participation, and argmax partitions. The explicit distance biases can create coherent cell-like regions without having identified physical interaction domains.

Use bounded diagnostics that test function:

1. **Geometry-only reference:** remove the learned content part of router logits in an evaluation-only counterfactual; compare routes and physical errors, not just colors.
2. **Fixed geometry, changed physics:** vary an admissible operating input in a model-side sensitivity experiment or compare genuinely paired stored cases. Measure whether detail retention and source response change. Do not claim physical correctness without solver targets.
3. **Fixed versus case-conditioned retention:** replace the controller input with a training-population summary during frozen evaluation. This tests whether adaptivity improves outcomes beyond a static contracted architecture.
4. **Restore closed detail:** at fixed weights, reopen selected closed nodes and measure task effects. This tests usefulness of removed distinctions inside that trained model; it is not a causal CFD conclusion.
5. **Permutation consistency:** permuting physical module ordering should preserve the physical prediction and corresponding set-valued organization after undoing the permutation.

All tests are descriptive/interventional checks of the surrogate. Group IDs are not physical objects. Do not prune a class just because it never dominates an argmax map. Conversely, do not call R<12 physically meaningful merely because training produced it.

## 13. Timing and computation accounting

On the same device and checkpoints, measure native and matched inner chunks separately (128 and 2048 where supported), Q8192, maps off. Retain preparation, prepared decode, full GPU forward and complete application as distinct scopes.

Use actual warmups and synchronized repeats, reversed order when useful, and report medians/dispersions. Measure:

- controller and T preparation;
- packing and source-control bank work;
- actual raw proposal logits evaluated;
- compact representation R and executed padded width;
- actual QM/QE fine and geometry rows;
- total and incremental allocation.

The immediate engineering expectation is a small controller/transform overhead rather than ~2x matched application time. Treat a substantial observed overhead as a profiling problem to solve within the pilot, not as an automatic consequence of a small matrix. Do not predict a speedup before measuring it, and do not attribute common larger-chunk gains to the new architecture.

If a compact path loses on a small real shape, keep the virtual reference and report the tradeoff. Physical sparse-row execution can be a later experiment; do not pay for a new kernel before the new structural mechanism earns its place.

## 14. Code touchpoints and cleanup boundaries

Inspect actual current code before editing. Expected reusable pieces at `427b24f`:

- `HONF_Proj/src/honf_forward_core/interface_fields/sparse_incidence_router.py`
- `.../sparse_incidence_group_control.py`
- `.../continuous_functional_coalescence.py`
- `.../coalesced_sparse_incidence.py`
- `.../group_control_pairwise.py`
- `HONF_Proj/src/honf_forward_core/organization/functional_fusion_tree.py`
- `.../organization/group_fusion.py`
- `.../interface_fields/core.py`, core configuration, runtime loader/registry.
- The physical wrapper's P0/P1/P2 preparation and live training-loss reduction.
- `HONF_Proj/tests/test_functional_fusion_tree.py` and existing quotient/physical-core tests.

Keep the new predictive path small: one reusable learned-detail organization module and one thin backend are sufficient in principle. Factor genuinely shared moment-bank/quotient helpers out of the accumulated experimental subclass chain if needed; preserve historical dispatch and loading. Do not introduce another large framework, broad lint rewrite, or mandatory solver dependency.

## 15. Repository and research-process rules

Read and obey the current `AGENTS.md` and `.githooks/pre-push`. Preserve the existing pre-push check, checkpoint trust/loading behavior, and other security measures.

Commit durable model code, ordinary maintained tests/configs, and the final textual research report. Keep one-time diagnostic runners, pilot outputs, checkpoints, raw arrays, generated figures and this response's standalone algebra support files in local ignored locations. Do not force-add them or hide prohibited artifacts behind renamed paths. Audit the entire outgoing commit range as required by the existing rule, push the working non-default branch, and verify remote/local tips match.

Use Git, the existing run allocator, ordinary configs and tests. Do not add hashes, immutable baseline copies, approval registries, monitoring daemons, or new security/contract gates for this experiment. Scientific stopping/selection follows actual measurements; preflight assertions do not replace execution.

No work in this ChatGPT review has launched training or changed the repository. This document directs the later Codex run.

## 16. Definition of a useful outcome

A positive result is not merely nonzero controller gradients or a lower expected R. It is:

- actual deterministic equal-function classes that vary usefully with the case/phase;
- preserved source-resolved physical information and exact compact/virtual equality;
- physically acceptable predictions across field and interface quantities;
- credible native design-boundary behavior;
- small, measured organization cost;
- a clear explanation of what did and did not improve.

A negative but useful result can be that active structural optimization finds only a static simplification, or that physically distinct functions cannot be tied without a significant task cost. Report that honestly. The previous passive threshold experiment did not establish either conclusion.

## 17. Sources and independent checks

### Project sources

- `HONF_Run1506_Continuous_Functional_Coalescence_Diagnostics.md`, especially “Structural prior and provenance,” e150/e500 formation, selected checkpoints, and execution tables.
- `organization/functional_fusion_tree.py`, particularly `_score_route_changes`, `node_contraction`, `build_tree_transform`, and `pack_exact_closed_subtrees`, at `427b24f`.
- `interface_fields/continuous_functional_coalescence.py::prepare`, `_route`, and diagnostic preparation, at the same commit.
- Existing v2-v4 source-moment and multiplicity tests.
- `AGENTS.md` and `.githooks/pre-push` at that commit.

### External methodological context

Louizos, Welling and Kingma, *Learning Sparse Neural Networks through L0 Regularization*, ICLR 2018, arXiv:1712.01312, especially equations 10-13. This supplies a differentiable stochastic mechanism for exact zeros and an analytic positivity probability. The source-moment quotient, conditional tree controller, ancestor-aware expected R, physical-loss coupling, and endpoint cubic in this plan are the proposed HONF construction, not empirical results from that paper.

Wen et al., *Learning Structured Sparsity in Deep Neural Networks*, NeurIPS 2016, arXiv:1608.03665. This is general precedent for actively learning structured sparsity; its speed results are not transferred to HONF.

### Checks executed for this proposal

The accompanying `algebra_check.py` and `algebra_results.json` are independent float64 CPU checks. They are not repository tests or a physical checkpoint replay. They verify the old flat-region zero gradient, new expected-R gradients, the Haar/product equivalence, expected-R counting against sampled partitions, compact/virtual nonlinear read and first-gradient parity, and a decreasing synthetic closure-bracket difference. A two-case toy is actually optimized to keep a necessary distinction and close an unnecessary one. It demonstrates a learning pathway only; it predicts neither HONF accuracy nor GPU speed.

Use these as reviewable raw material in local ignored paths, not as grounds to skip actual tests or to upload one-time experiment code.

### Executed independent-check summary

- Haar versus original nested projection product: maximum absolute difference 1.11e-16.
- Nonlinear compact/virtual toy output and selected first-gradient differences: 6.94e-18 and 3.47e-18.
- Expected R formula: 9.002423; 50,000 sampled partitions: mean 8.996960, Monte Carlo standard error 0.009478.
- Old keep-region derivatives were exactly zero at example scores 0.188, 0.230, and 0.730. The proposed expected-R gradient remained positive for every node in a clipped-open deterministic example.
- On an active two-function closure, reducing the synthetic logit half-bracket from 1e-2 to 1e-5 reduced the maximum field difference from about 4.74e-8 to 4.71e-14. These are not physical-coordinate magnitudes or trained-model results.
- A separately optimized two-case toy reached deterministic detail retention [0,1] for its redundant/necessary contrasts. This is evidence of an available optimization path, not a predicted result for ThermalChannel.
