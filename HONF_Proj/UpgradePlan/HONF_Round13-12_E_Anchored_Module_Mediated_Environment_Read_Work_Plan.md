# Round13-12: E-anchored, module-mediated environmental access

## Objective and authorization

Develop and empirically test the path **physical inputs → actual M/E interaction → locate the query's E → choose module organization views → obtain permitted original E members → query-conditioned reading → joint physical fields**. The intended improvement is a clearer, useful relationship between module choice and environmental access, with preserved physical accuracy. More E members, prettier groups or larger adjacency matrices are not objectives by themselves. Keep the successful lean anchored execution and individual member detail.

This document prepares an implementation and experiment task; writing it launches no training. Execute on the current `agent/honf-core-next` branch, using `3ecc57a` and subsequent user work as the starting checkout. Preserve existing edits, historical S/X runs, checkpoints, aliases and unrelated jobs. Do not reset the checkout, reinterpret an old checkpoint as the new architecture or change an active run.

The user explicitly authorizes mature segmented comparisons up to **1000 epochs per declared candidate** and **one or two separately named, fresh TRAIN600 formal candidates launched directly toward 1000 epochs each**. The bounded matrix below governs how many candidates may be fitted. Formal100 is not a mandatory decision boundary in this round. After each formal launch, confirm actual startup once, then finish the task: **no Luna formal monitor, observer, polling, completion wait or follow-up automation**. Ordinary in-process validation, health reporting, curves and checkpoints remain enabled. Later formal evaluation commands must be delivered; their unobserved results must not be invented.

Use physical GPUs 0 and 1 concurrently when available for independent segmented candidates, and later for the selected formal candidates. Preserve other jobs; do not preempt S/X or silently substitute a requested device. Use ordinary existing device/process checks and launch facilities, not a new reservation or identity-lock service. This round-specific authorization supersedes the usual segmented500 ceiling and initial formal100 review, only within the explicit matrix and horizons here.

## Evidence to retain and lessons to carry forward

Read the [R13-11 report](../docs/reports/HONF_Round13-11_Stable_EE_Relations_and_Geometric_Organization_Report.md), its [work plan](HONF_Round13-11_Stable_EE_Relations_and_Geometric_Organization_Work_Plan.md), the [Thermal protocol](../docs/guides/Thermal_Model_Development_Protocol.md), and the [R13-10 closure](../docs/reports/HONF_Round13-10_Useful_Environment_Coverage_Report.md). The fixed [formal S/X epoch1600 access review](../diagnostics/round13_11_20261028/side_visual_review_20261029_0645/README.md) and [X hypergraph/decoder review](../diagnostics/round13_11_20261028/x_classical_hypergraph_and_decoder_review_20261029/README.md) supply concrete mechanism examples, not complete VAL89 predictor comparisons or a statement of current live training progress.

S is the primary design and fidelity reference. X demonstrates that retained nonself E use and a broader directory can coexist, while also illustrating common remote suppliers. In the fixed VAL0678 epoch1600 scene, S's actual final M←E rows have 1–88 positive E members; X has seven empty M rows and only three nonempty rows, with 55/3/1 E. The TRAIN0003/M1 scene has 142/75 positive M←E members for S/X respectively. A module-mediated route therefore offers a real alternative organization, but its width and usefulness must be measured rather than assumed.

A2-D established the value of contextual exchange; U-H established reusable interaction-derived organization and substantial access redundancy. R13-3's fixed-zero null plus hard sparse training selector exposed an avoidable dead learning path. Relative restored continuous member learning. R13-5 retained organization and original member detail together. R13-8 removed expensive global entrance search through E anchoring. R13-11 separated EE gate width/amplitude and exposed late concentration. Preserve these advances. Do not return to compulsory pooled features, unorganized raw-value bypasses, decoder programs that repeatedly score the same decision, or auxiliary constraints designed only to keep E counts high.

Three distinctions govern this round. Sparse E access is different from local E access. Zero M values read is different from zero M organization rows consulted. An actually executed directed relationship is different from a new composition of several such relationships. Make these distinctions visible in code traces, equations and figures.

## Controlled implementation boundary

Keep S's physical token ownership, generic joint-field head, two contextual cycles and their MM→EM→EE→ME order, relative-quadratic EE gate, initial width, contextual projections, physical objectives and native adapters. Retain A as the primary locator and preserve L's completed-row blending and loading behavior. Do not add a separate temperature/flow head, heat-affine correction, heat-independent routing assumption, compulsory M contribution, local-neighbor quota or physics-specific decoder.

The implementation seam is the prepared organizer plus packed membership/read planning in [anchored.py](../src/honf_forward_core/interaction_reused_member_routing/anchored.py). Add an explicit opt-in architecture/configuration for the new organizer, preferably in a small adjacent module; keep historical S/X defaults and checkpoint loading unchanged. Reuse the existing raw scorers, prepared normalized keys, typed read-gate modules, value projections, packed reductions, query tiling and shared paired-mode score plan. Do not enable the entire legacy directory merely to retrieve one relation block.

Capture **actual final-cycle EM, EE and ME** with `_safe_snapshot_gate`, original donor/receiver IDs and stage identities. EM means E←M; ME means M←E. Obtain ME from the actual M-receiver stage, not the transpose of EM. EM precedes EE/ME, and the final contextual M/E values have their respective later versions. Their composed access law is a routing composition of stage-labelled relations, not a same-time physical influence matrix. The new method must genuinely reuse these live relationships rather than learn a disconnected directory scorer.

## One common routing law

Let a be the query's located E. Start from the **full**, pre-truncation applied relations, the existing separate structural self ledger S, and S's original prepared measures:

$$
d_{ai}=(G^{EE}_{ai}+S_{ai})\mu_i^E,\qquad u_{am}=G^{EM}_{am}\mu_m^M,\qquad b_{mi}=G^{ME}_{mi}\mu_i^E.
$$

Here S contains the explicit own-E access, not learned EE self. Define direct environmental route mass and conditional destinations by:

$$
d_a=\sum_i d_{ai},\qquad D_{ai}=d_{ai}/d_a,\qquad B_{mi}=\begin{cases}b_{mi}/\sum_jb_{mj},&\sum_jb_{mj}>0,\\0,&\text{otherwise}.\end{cases}
$$

The structural-own ledger makes the valid anchor's direct route nonempty. A genuinely empty ME row stays empty. Apply physical measure once at each declared conditional stage; B is already measure-weighted and must not acquire a second donor-measure factor during composition. Source masses and B have different meanings: the former determine which route to call, while B describes where that selected route leads. Row normalization does not turn B into the original ME message amplitude. Record raw ME row mass and whether normalizing a weak row amplifies an unhelpful destination distribution.

For each located E, the small route alternatives are **one real direct-E route plus its EM-related M records**. Compute raw scores with the existing q–ownE and q–M member MLPs, using contextual keys and native relative geometry. Calibrate this small source set once with the inherited prior-weighted bounded law and source priors `[d_a,u_a1,…]`, then form:

$$
\alpha(q,a)=\operatorname{softmax}\!\left(\log[d_a,u_{a1},\ldots]+\ell_{\rm source}(q,a)\right).
$$

The first alternative is a real, scored E organization that can provide information. It is neither an empty null nor a fixed-zero baseline. There is no prescribed 50/50 role split or new controller network. Ineligible/padded M are excluded. Use the live source weights in the environmental prior:

$$
w_i^E(q,a)=\alpha_0(q,a)D_{ai}+\sum_m\alpha_m(q,a)B_{mi}.
$$

Only then pass original E identities, coordinates, keys/values and this organized prior to the existing query-conditioned E reader. A weighted composition of directory rows is not pooling E features. Each original E value remains individually available until the final query read. Empty ME rows contribute zero environmental mass; retain their selected source mass in diagnostics rather than invent a replacement relation or renormalize them into fictitious destinations.

Reuse the module source decision for the module value read:

$$
p_M(m\mid q,a)=\frac{\alpha_m}{\sum_{r\in M}\alpha_r},\qquad c_M=\rho_M\sum_m p_M(m\mid q,a)V_Mh_m.
$$

The formula uses the retained M support in restricted execution and all eligible M in full execution, returning exact zero context when that support is empty. Reuse the source-calibrated module scores and conditional probabilities to compute the existing strength statistics. **Do not multiply the same M score into these weights a second time.** The existing smooth value strength rho_M controls whether M values matter; it must not multiply or switch off the M-mediated E prior. Source selection and value amplitude have separate semantics even when they share raw member scores.

The own-E raw score may participate as the real source-alternative score and as that original member's final E score. Cache its raw evaluation once; document these two hierarchical uses rather than pretending they are one probability or accidentally applying an extra likelihood factor. Track alpha0 against final own-E beta on the fixed panel so a possible own-E reinforcement loop is visible: caching removes duplicate computation, not this statistical effect. Full/restricted branches share raw scores, while their E candidate-dependent calibration, member probabilities and strength statistics are recomputed for their actual supports. This new M weighting law is not claimed to be numerically equivalent to S's old role-conditioned reader.

Keep the final E reading law and physical head structurally intact:

$$
p_E(i\mid q,a)\propto\mathbf1_{i\in C_E}\,w_i^E(q,a)\exp\ell_E(q,i),\qquad c_E=\rho_E\sum_i p_E(i\mid q,a)V_Eh_i,\qquad \widehat{\mathbf y}=F([z_q,c_M,c_E]).
$$

For L, complete each located anchor's conditional read before applying the inherited locator-weight blend. Share/coalesce repeated original members without merging incompatible per-anchor normalizers. Do not collapse all located rows into one prior as a shortcut.

## Two declared architecture candidates

**P — relation-composed access.** Use the actual normalized ME destination rows B unchanged. This is the primary test of whether selecting module organization views improves E access and physical fidelity. It deliberately has no added geometry bias. With one M, its M-conditional destination row alone is constant; the direct-E alternative and final q–E scores can still vary. Do not describe that as established early regional candidate formation unless actual pre-score member identities vary.

**G — anchor-conditioned relation composition.** Change only the conditional ME destination rows:

$$
B^a_{mi}=\frac{b_{mi}\exp\bar b_{\rm geo}(a,i)}{\sum_j b_{mj}\exp\bar b_{\rm geo}(a,j)}
$$

for nonempty rows, with exact zero otherwise. Prefer a clearly specified mean across the existing final-EE geometry-bias heads, captured before content/gate sparsification. This reuses an existing geometry function; it is a new address-conditioned routing composition, not an additional exchange that allegedly happened in the encoder. Preserve its live gradient, rather than consuming a detached inspection snapshot, and state its head aggregation explicitly. Apply this factor once. Use stable log normalization without inventing positive relations or forcing a near-neighbor support. A zero ME relationship stays zero; a distant positive relationship remains legal.

G makes the destination distribution depend on the located E before expensive q–E scoring, including M1. Fine variation within one E coverage region remains the final reader's job; L can smoothly mix nearby anchor-conditioned reads. The geometric bias is an explicit prior and must be shown separately from learned relation support. Inspect whether the inherited geometry kernel actually provides useful regional differentiation; a broad or distant learned kernel is not silently replaced with a hand-tuned radius.

If a demonstrated geometry-kernel defect prevents G from answering its hypothesis, one simple adapter-scaled radial kernel may replace that kernel as a named, documented revision within the existing G budget. Do not add it on top, fit a third permanent architecture arm or introduce neighborhood rewards. Similarly, a demonstrated weak-ME-row amplification problem may justify one measured row-mass alternative within a candidate's remaining budget; do not pre-emptively add confidence networks or a grid of calibrations.

## Restriction, zero M and learning stability

The full branch uses continuous source weights and all positive composed E destinations. Its E-task gradient must reach the live module-source mixture and actual relations wherever mathematically nondegenerate. Selecting M only through a detached discrete membership would lose that learning path. Keep the inherited shared full/restricted physical objective, with no new coverage/entropy/locality losses or forced opening/leaf phase.

Compute source calibration and alpha once over the full eligible source set. Restricted execution masks omitted M contributions without recomputing the source softmax or renormalizing its remaining source weights. The restricted source prefix may select the direct-E source and no M. Always append/retain the real direct-E source with its **live alpha0**, as well as the original own-E member with positive composed access. If no M is selected, return zero M context without gathering M values; the same E reader still operates through D. This is legal E-only reading, not a separate fallback predictor. Preserve representable positive own-E mass with stable log-mixture/normalization and the inherited positive-support numerical handling, including high-contrast native FP32 cases; do not merely append an index whose prior underflowed to zero or invent an epsilon ME relation. Distinguish a source-prefix decision to consult no M, small M value strength, and empty physical M inventory in every trace and test.

Before substantive fitting, execute four same-weight modes on a small fixed native panel: continuous sources/all E; continuous sources/E95; source95/all E; source95/E95. This isolates source approximation from member approximation without creating four training arms. Start with the continuous reference and single-restriction modes before testing the combined executor. Each mode's E95 prefix uses its actual composed environmental weights after any source masking; compare omitted path mass with the full composition separately. The planned paired candidate uses the continuous branch and the combined restricted branch after actual execution/gradient qualification; retain a simpler restriction if the combined one has a concrete diagnosed cost or quality defect, and apply that choice consistently to P/G before their comparison. If later diagnosis changes restriction in only one arm, label the additional equation change and use same-weight execution comparisons to separate it from the geometry hypothesis.

Two 95% prefixes are **not** a single 95% guarantee. Record source alpha mass retained, conditional M mass retained, full composed E mass, mass lost through omitted M routes, mass assigned to empty ME rows, and mass retained after E-member restriction. In particular, report final retained E path mass relative to the full composed E mass; do not assume 0.95 squared is exact when empty rows exist. These are execution-approximation measurements, not physical-error bounds or mandatory organization targets. The selected formal configuration must explicitly name both coverage settings; source and E coverage may remain distinct rather than being hidden under the old `joint95` label.

All E candidate assembly and coverage selection must precede expensive detailed E scores in standalone restricted inference. The only E raw score prepaid for route choice is the located own-E score. Query M/source work, live relation composition, row lookup, coalescing, selection and index construction all count as work. Do not materialize a persistent dense batch×E×M×E tensor. Use packed actual ME edges and shared/lazy unique `(anchor,M)` computations; queries sharing an anchor reuse address conditioning and row indices. Paired training may pay the full branch's score union; do not rename that as sparse-training savings.

## Execution qualification and focused tests

Extend the existing [packed anchored tests](../tests/test_e_anchored_packed_execution.py), [anchored helpers](../tests/test_r13_8_anchored_core_helpers.py) and [R13-11 gate fixtures](../tests/test_r13_11_relative_ee.py) where appropriate. Do not create a new validation framework or a suite whose cases merely repeat the implementation.

Use one small explicit reference for directed route composition with nontrivial original IDs, padded M, zero-area E and overlapping ME destinations. Check EM versus genuine ME, physical measure exactly once, structural own access separate from learned self, exact empty rows, actual alpha/D/B composition and duplicate-member coalescing. Verify the dense mathematical reference against packed execution for fields and input/parameter gradients in high precision, then run actual native FP32 prepare/decode/backward, activation replay, save/reload and same-lineage resume. S's unchanged configuration must still load and execute.

In an M≥2, nondegenerate fixture, demonstrate native E-reading loss sensitivity to module-source weights and ME destinations through the continuous path. In M1, the derivative of a normalized module-only conditional probability is zero by identity; do not mislabel that expected result as a broken route. Test two or more spatially separated own-E addresses for **pre-score** E candidate differences in G, and separately test final E weights. Use actual geometry/relations as well as a tiny constructed correctness fixture; the latter does not establish a trained regional organizer.

Exercise zero selected M, empty physical M inventory where supported by the core fixture, empty selected ME rows, all-zero learned EE, own-E-only access, remote legal donors and no nonself E. Test zero M value contribution with retained M-directory use through an explicit read-strength intervention/fixture: the finite-logit smooth gate generally remains positive, so report learned near-zero strength continuously rather than adding a new hard skip. Finite outputs and a positive legal own-E read must survive. Test that rho_M changes M value amplitude without erasing the E routing law. Exact zero M directory use must give exact zero M value gather/context in that mode.

Run native A and L paths, including original fields and first spatial derivatives around locator boundaries, and saved/reloaded execution. Reuse the existing heterogeneous-E/irregular-measure adapter fixture: object features and controls remain owned by their input records, changing legitimate E inputs can affect predictions, and the joint predictor accepts native field dimensions. This is interface and learning-path evidence, not a mature cross-dataset accuracy claim. No new Wind/electro-thermal fitting or simulator campaign is included.

## Segmented experiment matrix and budget

Reuse the exact completed S control at `/data2/wanglz/ModularDT/round13_11_20261028/train150/thermal_r13_11s_e_anchor_train150_v1`. Its literal 100/200/…/1000 checkpoints and resolved configuration are available. Reuse the fixed TRAIN150/VAL22 manifest, seed0, sampling recipe, Q1024, effective batch48, FP32, constant LR0.0003, native objectives, paired full/restricted weights and common initialization. Preserve the common train-only normalizer. The mature S formal1600 examples are mechanism references; they are not substituted for matched-age TRAIN150 fidelity controls.

| Lineage | Training allowance | Role |
| --- | ---: | --- |
| Existing S | No new fitting by default | Matched-age baseline from its literal saved checkpoints. |
| P | At most 1000 TRAIN150 epochs | Module-mediated relation composition. |
| G | At most 1000 TRAIN150 epochs | Same law, changing only early anchor geometry conditioning. |
| Necessary repairs | Charged within the combined P/G allowance | No extra fitting pool, refreshed allowance or unbounded third portfolio. |
| Fresh S, only if actual common-recipe mismatch prevents reuse | At most 1000 TRAIN150 epochs | Conditional control; explain the concrete mismatch before spending. |

The hard new segmented fitting allowance is **2000 epochs when S is reused, or 3000 only if that necessary fresh S control is run**. Every retry, discarded partial epoch and diagnostic fit is charged. Keep some of the remaining allocation available for demonstrated repairs instead of automatically spending both endpoints first. A repair may reduce a candidate's final age; report matched saved ages and the actual expenditure rather than hiding a reset. No individual candidate lineage exceeds1000. A zero-fit replay/evaluation does not consume a fitting epoch, but its compute is still reported.

Initialize new P/G fitting from the same scratch common tensors and seed as S, not from a mature S checkpoint with undeclared extra training. Preserve equal initial encoder/scorer/gate/head tensors and label every new or removed parameter. Existing S checkpoints may support zero-fit counterfactual directory probes, but those probes are not silently promoted into formal initializations. Avoid inherited inactive schedule fields becoming new forced training phases.

Use only a short initial segment, normally no more than 10 epochs, to verify the real path, learning health and checkpoint/resume; this is an execution review, not an architectural verdict or a model warmup rule. Charge it to that candidate's allowance and resume its unchanged law if healthy. Review substantive saved trajectories at 200/500/1000, with matching S checkpoints. Do not force an extra100-epoch stop when nothing needs diagnosis, and do not interpret a healthy early accuracy deficit as failure. A1000 subset endpoint remains segmented evidence; full-data fidelity is evaluated only from actual full-data runs.

## Active diagnosis and useful decisions

Pause the affected candidate promptly for numerical failure, wrong original IDs/stage routing, replay/resume corruption, an accidentally detached continuous mixture, or loss of native physical-input dependence while a coordinate-only predictor keeps lowering loss. Preserve the failed state and a compact explanation, isolate one cause, repair within budget and verify actual execution before resuming. Do not continue a known disconnected reader simply because its total loss declines. Reuse existing native health outputs and failure reporting; do not invent a monitoring framework or heuristic admission service.

A narrow row, one query with no M/nonself E, or legitimate empty ME is not automatically a collapse. If module-mediated contribution disappears broadly, inspect source priors and score scales, actual ME row emptiness/mass, source-prefix omission, conditional B amplification, geometry conditioning, final E probabilities and typed strengths separately. Compare selected source mass on empty/weak ME rows with those modules' value contributions: a useful M value does not require a nonempty E directory and must not be deleted simply for lacking one. Inspect native termwise gradients only on a small fixed TRAIN panel when needed. Do not freeze gates, force a minimum degree, reward entropy, fill missing links or compensate with a new physical branch.

P may establish useful composition without early regionality; G may improve candidate differentiation while costing more or harming tails. Preserve these distinct findings. If the inherited geometry kernel is ineffective, diagnose it rather than presenting later q-dependent weights as an early-candidate success. Use the four same-weight restriction modes to locate whether a deficit begins in the new full law, M restriction or E restriction before changing another equation.

## Scientific reviews: fidelity, access and useful organization

Use the declared VAL22 for segmented statistics and the existing input-selected detailed scenes0274/0293/0678. Keep all nine native mean and p90 errors, including material/surface temperature and original-module peak tails; emphasize a small field-error comparison and a few reference/prediction/residual panels. Training loss is a health/trajectory view, not the substitute for native errors. Do not redraw the evaluation membership from current errors or introduce response-fitting differences between arms.

For access, reuse TRAIN0003/M1 and VAL0678/M10 with the same saved512 original query IDs and geometry-fixed detailed locations when applicable. **M1 has no canonical held-out VAL cases**: its TRAIN illustration demonstrates routing behavior, not held-out M1 fidelity. Show actual original M, own E and nonself E reads, their counts and raw beta contributions, zero-M-source and zero-M-value frequencies, and pre-score candidate identities. Separate available relation support, admitted candidates, executed records and final weight concentration. Raw beta need not sum to one, and its share is not a causal output attribution.

Trace both contextual cycles on the small panel, and preserve final EM/ME/EE gate provenance through source alpha, composed E prior, retained sets and query beta. Show collective/remote service and shared suppliers as well as local organization. Use three complementary graphics: original-domain prepare/read maps; conventional enclosing hyperedges with shared original nodes and an actual-read companion; and a compact member/source incidence view. Derived routing compositions and explicit geometry bias receive separate labels. Do not draw a spatial partition where only overlapping hub service exists.

At500/1000 on the fixed panel, remove M-mediated E paths with contexts held fixed and recompute the legal remaining E read; separately suppress M values without removing their environmental routes. If needed, remove actual ME/EE exchange and re-prepare as a separate intervention. Include prediction/native-error changes and one collective set-versus-single-member probe for a selected mature candidate. Use existing saved physical responses/controls when reference identities match; retain zero-control temperature response as relevant, and **do not add the Thermal-specific heat-to-flow invariance test**. No intervention is renamed physical causality, and not every edge removal must hurt every query.

Select by preserved field fidelity and useful organized access together. Do not demand improvement in all nine metrics, a larger E count for every query, a fixed group count, hard locality or a block-diagonal matrix. Do not stop exploring because the first short fit is worse. A complementary second formal candidate needs a clear accuracy/access or regionality tradeoff; it is not launched merely to fill a slot.

## Measured work and implementation economy

Measure one same-device native low/high-M complete inference panel and a paired training forward/backward panel at a mature endpoint, with preparation, directory conditioning, source choice, E candidate construction, detailed scores, values and extraction included. Report allocated baseline/peak/increment and reserved memory separately. Component profiling is used only to explain a measured bottleneck; overlapping spans are not added into a fictitious total. Reuse comparable S workload definitions and do not benchmark against concurrent training on the same GPU.

P/G source choice adds sequential work. A larger ME union can increase E score count; geometry preparation may offset later savings or may not. Compare all/full and actual restricted standalone execution, including all indexing costs. Shared paired training scores do not establish standalone inference speed. Retain proven projection/key/score reuse and fix clear duplicate computations, but do not combine a wholesale decoder rewrite or new attention family with the organizer comparison.

## Monitoring: segmented reviews, formal handoff

For every segmented launch/resume, delegate read-only monitoring to an explicitly selected inexpensive `gpt-6-luna` agent with `fork_turns="none"` and only the owned run path, physical GPU, next epoch review, histories/curves/checkpoints and actionable notification conditions. One startup inspection estimates actual epoch time. The next inspection is near the declared short stop or 200/500/1000 review, normally after 80–90% of estimated remaining duration. Luna stays quiet until completion, failure, meaningful delay or required user action. The parent handles independent work or rests; it does not duplicate polling. Use existing sessions/waits, not a daemon or watch service.

Final formal training is different by explicit user instruction. After selection, launch **one or two fresh TRAIN600 lineages directly to 1000 epochs**. Keep common scratch initialization, full-TRAIN-only normalization and all unchanged scientific objectives. Use separately named P/G configs, for example `thermal_r13_12p_e_anchor_mediated_train600_formal_v1` and `thermal_r13_12g_e_anchor_conditioned_mediated_train600_formal_v1`, recording the actually selected equations and both coverage settings. The maximum formal fitting authorization is 2000 epochs across two candidates; it is not permission to fit controls or additional alternatives.

Confirm each formal startup once from actual configuration/first progress, ensuring its intended lineage, device and1000 stop are active, and then end the task. **Do not attach a Luna monitor or observer, poll later epochs, wait for formal 100/1000, schedule an evaluation or automate a restart.** Existing in-process numerical failure handling, regular validation/history/curves and100-epoch checkpoints continue normally. Deliver tested manual evaluation/resume commands for the selected scientific milestones, including matched-age S comparison, and describe the launch as startup only. No post-1000 extension is authorized here. If resources are unavailable, preserve a tested concrete launch command and explain the resource conflict rather than interfering with another job or claiming an unlaunched run started.

## Storage, upload and completion

Before any experiment payload, verify DATA2 mount and the declared workspace symlinks. Use `/data2/wanglz/ModularDT/round13_12_<date>/` with a round diagnostics alias. The implementation round's hard storage envelope is **24 GiB retained plus 4 GiB temporary**, including checkpoints, native numerical evidence, curves, graphs and caches, with at least 100 GiB DATA2 and 20 GiB home headroom. Record actual free space and peak estimates; reduce exports rather than duplicate datasets or silently use home storage. Reuse exact existing shared caches and physical-reference arrays.

TRAIN150 payloads remain under the round's ignored diagnostics root. Formal payloads may live under its `formal/`, but their aliases must be exactly beneath `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/<run_name>`. Never put aliases directly under `Trained_Results` or rewrite historical payloads. Retain latest/best-field states and the existing100-epoch milestones, initial/final and10-epoch refreshed loss curves, full histories and actual fixed-panel evidence. Do not create extra 25-epoch snapshot copies or per-review cache stores.

Deliver the English R13-12 report, durable opt-in implementation/configs and proportionate tests, a short figure index with inspected PDF masters/relative PNG embeds, actual per-arm expenditure, selected formal rationale, startup receipts and tested user-controlled evaluation/resume commands. Keep prose/captions on single source lines. State completed segmented outcomes separately from formal startup and unobserved full-data outcomes. Cleanup only verified round-owned disposable data after exact allowlist/consumer/dependency checks; retain scientific checkpoints and the receipt of removed/retained bytes and actual remaining capacity.

Commit/push durable changes on the current non-default branch through existing safeguards, preserving unrelated edits. Configure `core.hooksPath=.githooks`, audit the entire outgoing range and artifact list, then verify local/remote tips match. Generated figures, one-time diagnostics/renderers, arrays and runs remain ignored on DATA2. Do not add new hashes, freezes, baseline services, blocking research gates, speculative compatibility layers or monitoring infrastructure. Ordinary Git history, existing IDs, focused tests, atomic writes and actual execution are the tools for this task.

## Concise execution prompt

Execute `HONF_Proj/UpgradePlan/HONF_Round13-12_E_Anchored_Module_Mediated_Environment_Read_Work_Plan.md` on the current `agent/honf-core-next`, preserving user edits and S/X jobs. Reuse the matched S TRAIN150 control. Implement P/G with actual EM/ME/EE provenance, a real direct-E source, continuous live module-to-E routing, independent M value strength, original-member reads, shared raw scores and the existing joint head; G changes only early anchor geometry conditioning. Verify M1 pre-score regionality, zero-M semantics, empty rows, native gradients/replay/resume, A/L and heterogeneous-input compatibility. Fit only the declared segmented candidates within the 1000-epoch-per-candidate and combined budget, using sparse Luna ETA monitoring and active diagnosis rather than premature failure verdicts. Select one or two useful candidates, launch fresh TRAIN600 formal runs directly toward1000 epochs on available physical GPUs0/1, verify startup once and finish with no formal monitor or wait. Deliver measured fidelity/access/cost, clear hypergraph figures, tested commands, cleanup and the audited commit/push.
