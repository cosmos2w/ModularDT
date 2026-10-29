# HONF Round13-7: One Organization-Conditioned Distribution for Member Reading

Prepared 27 October 2026 from the current branch containing `e9a5ba8`, the completed R13-6 report, its actual implementation, and a bounded read-only inspection of the ongoing formal lineage. This document authorizes the described experiments when execution of this plan is requested; preparing it launches none. Preserve subsequent user edits, active jobs, historical loaders and all original scientific checkpoints.

## Purpose: simplify the successful route, one decision at a time

R13-5 established the working backbone: actual exchanges form reusable M/E organization views; query-dependent proposals and those views identify eligible original members before expensive reading; contextual member identity, position, measures and controls survive through one joint physical predictor. R13-6 made the composition of organization rows affect the final member distribution and recovered meaningful training execution savings. R13-7 should retain that progress while removing a parallel read law that was deliberately retained for the preceding controlled experiment.

The single scientific question is: **should the strength of a role's read be computed from the same organization-conditioned member distribution that actually supplies its information?** First implement equation-preserving execution cleanup, then change only that dependency. Do not combine it with a new score calibration, cutoff, relationship measure, scorer, physics objective, summary bottleneck or organization-merging policy.

The [R13-6 report](../../../docs/reports/_bk/20261029_R13_Consolidation/20261027_210114Z_HONF_Round13-6_Faithful_Integration_and_Useful_Organization_Report.md), [R13-5 report and mature comparison](../../../docs/reports/_bk/20261029_R13_Consolidation/20261027_103711Z_HONF_Round13-5_Joint_ME_Organization_and_Member_Routing_Report.md), [current faithful reader](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/faithful.py), [typed organization operator](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/typed_organization.py), [H distribution dispatch](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/typed_organization_core.py), [shared execution implementation](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/consistent.py), and [Thermal development protocol](../../../docs/guides/Thermal_Model_Development_Protocol.md) are the primary sources. Three independent reviews covered equation semantics, execution structure and scientific evidence; their recommendations were reconciled around this narrow change rather than added as separate experiment arms.

## What R13-6 accomplished, and what the saved later history says

R13-6 O preserves original ME fields, native observables and live gradients at measured FP32 agreement while reducing the prepared 48-case paired forward/native-loss/backward median from 3.174 to 1.729 seconds, or 45.5%. That fixed panel is heavily M1-weighted and excludes fresh sampling and parameter application. Actual qualification timing supplies separate whole-epoch evidence. Complete native inference did not improve, so reuse the equivalent training work without relabeling it as an inference speedup.

Scratch H100 improves seven of nine canonical full-native VAL89 means against ME100. Fluid/surface means worsen by 5.50%/4.26%, and their worst errors worsen by 25.48%/24.28%. H skips 89.51% of the full reader's expensive member-score rows, but still executes all-address row contractions and adds 6.1–9.3% same-state complete inference latency over O on the measured panel. Its fixed-marginal finite-dose organization control changes normalized joint fields by up to 0.05035, versus O's 1.25e-6. This is an actual operational grouping effect beyond member marginals; retain it as a positive result while assessing native fidelity separately.

The one initial status inspection in this planning session observed recorded epoch 415, retained epoch 400 and an owned continuation process on physical GPU 1. No epoch-500 checkpoint was observed in that inspection. No observer, wait or polling was attached. This is a dated status observation, not a claim about the run's state at later delivery or execution of this plan.

Analysis of the fixed saved epochs 381–400, with matching case order and query draws, finds all eight mean sampled TRAIN loss terms lower for H than original ME by 5.21–16.20%. Saved engine validation at epoch 400 provides a more informative separation:

| Saved epoch-400 observable | H versus ME error change |
| --- | ---: |
| u / v / p / vorticity mean RMSE | −4.67% / −2.44% / −1.53% / −4.67% |
| Fluid / surface / material temperature mean RMSE | +2.59% / +11.78% / +13.72% |
| Composite engine field score | −0.46% |

These are fixed sampled-native, deployed-restricted engine metrics on canonical VAL89: 1,024 fluid receivers, 32 material receivers per module and the maintained surface stride. RMSEs average cases equally; surface/material case MSE first averages valid receiver errors within each module, then averages modules equally before taking the square root. The composite averages standardized flow and temperature MSE groups and excludes q-proxy and peaks. Its small decrease therefore coexists with meaningful temperature regressions. Do not present these saved measurements as a fresh complete-native nine-observable evaluation. [Read-only planning evidence](../../../diagnostics/round13_7_planning_20261027/planning_evidence.json) records the exact comparison and source paths. The [maintained validation implementation](../../../Case_ThermalChannel/src/channelthermal/training/content_hypergraph_task.py) and [restricted prediction dispatch](../../../Case_ThermalChannel/src/channelthermal/training/joint_member_routing_task.py) define this panel.

This evidence supports continued work and a specific fidelity investigation, not a return to premature pooling or a claim that similar loss curves mean equivalent fields. Neither the temperature difference nor row overlap is already attributed to the gate-statistic split. R13-7 must measure that proposed dependency change rather than assert it solves the remaining errors.

## Current equations and the one proposed scientific change

Use a for an organization row, i for an original member, and t for input role M or E. The actual applied directed exchange ledger H, its stage provenance and explicit M identity access remain unchanged. The prepared retrieval view is R_ai = H_ai μ_i / Σ_j H_aj μ_j, with the current role measures and whole-row M+E normalization. Query view proposals w_a produce the early member prior P = Normalize_legal(wR), preserving the implemented final legal-support normalization and exact-zero empty behavior. A single joint95 prefix C selects members before expensive member scoring. It may omit either role.

Keep the inherited candidate-conditioned likelihood calibration ell_i. For the original marginal reader, the role distribution is

\[
p^O_{i|t}=\frac{\mathbf1_{i\in C_t}P_i e^{\ell_i}}{\sum_{j\in C_t}P_j e^{\ell_j}}.
\]

H instead keeps competition inside each organization view before coalescing onto original IDs:

\[
d_{at}=\sum_{i\in C_t}R_{ai},\quad L_t=\sum_a w_a d_{at},\quad Z_{at}=\sum_{i\in C_t}R_{ai}e^{\ell_i},
\]

\[
p^H_{i|t}=\mathbf1_{i\in C_t}\sum_a
\underbrace{\frac{w_a d_{at}}{L_t}}_{\text{query-dependent view weight}}
\underbrace{\frac{R_{ai}e^{\ell_i}}{Z_{at}}}_{\text{competition inside that view}}.
\]

Empty terms return zero. This is a probability over original members within each nonempty selected role, not a feature replacing those members. R13-6 H currently reads mu_t = Σ_i p^H_{i|t} V_i, but computes strength from p^O:

\[
\rho_t^O=g_t\!\left(\widehat q,\sum_i p^O_{i|t}\ell_i,
-\frac{\sum_i p^O_{i|t}\log p^O_{i|t}}{\log N_t}\right),
\qquad c_t^{13\text{-}6}=\rho_t^O\mu_t.
\]

Here g_t is the existing typed strength MLP followed by the current bounded continuous contrast-to-strength function, and N_t is the existing full-scene eligible role count. Normalized entropy is exactly zero for N_t≤1; other rows retain the current clamped-log convention for zero probabilities. Strength includes the existing selected-role-nonempty and query-valid masks, and all zero-mass role distributions return exact zero. This split was a valid controlled design: R13-6 could isolate value redistribution while holding strength semantics fixed. It is now a concrete target for simplification, not evidence that the preceding implementation was invalid.

Candidate U uses **one final H distribution for both statistics and values**:

\[
\rho_t^H=g_t\!\left(\widehat q,\sum_i p^H_{i|t}\ell_i,
-\frac{\sum_i p^H_{i|t}\log p^H_{i|t}}{\log N_t}\right),
\qquad c_t^{U}=\rho_t^H\sum_i p^H_{i|t}V_i.
\]

Keep the same g_t parameters, two-logit contrast, bounded continuous strength, full-scene entropy denominator, empty/query masks, independent M/E strengths and joint physical head. The H probabilities and statistics remain live autograd tensors. No cross-role amplitude budget is introduced. Do not remove strength entirely or turn its two outputs into a different parameterization during this comparison.

At identical upstream state, U and historical H retain the same w, R, P, C, raw member scores, ell, p^H and member values. Only the statistics supplying strength change. The local identity

\[
c_t^U-c_t^{13\text{-}6}=(\rho_t^H-\rho_t^O)\mu_t
\]

isolates the same-state context change; the nonlinear joint head need not satisfy a linear output identity. U is a changed model, not an equivalent execution rewrite. Its strength gradient is newly coupled to the distribution actually read: dc_t = rho_t dmu_t + mu_t drho_t. That is a clear simplification of meaning and a testable learning change, not an automatic accuracy or stability guarantee.

## Give each operation one explicit job

| Existing quantity or operation | Purpose in the R13-7 path | Decision |
| --- | --- | --- |
| Applied H, member ownership/measures and stage provenance | Scene-reusable organization of original records | Preserve |
| Whole-row R and query view proposal w | Access views and their query relevance | Preserve |
| Marginal P and joint95 candidates C | Early inexpensive admission before member scoring | Preserve; do not reinterpret 95% prior as 95% physical information |
| Candidate-conditioned ell calibration | Keep the established member-score scale | Preserve in this round |
| Per-view H competition and coalesced p^H | Final relative information allocation | Compute once per role/read |
| Existing typed strength rho | Amount of that role's actual read | In U, derive statistics from p^H |
| Parallel p^O for H strength | Previous controlled-isolation mechanism | Remove from U ordinary prediction |
| Inherited A joint-M/E likelihood | Used by historical A/B, discarded by O/H prediction | Skip when unused; preserve legacy/diagnostic meanings |
| Final H numerical normalization and empty handling | Stable FP32 unit-role behavior | Preserve |

The resulting path is: owning physical records → actual exchanges and reusable H/R → query w/P → joint95 original-member candidates → selected-member scores and H competition → one distribution for read statistics and values → joint physical field. P and p^H have different jobs and should not be forced into one probability merely to reduce the number of symbols. A query-controlled final weighted reduction is legitimate information integration; it does not recreate a pre-query pooled bottleneck.

## Stage E: equivalent execution and clearer internal structure

Use the existing prepared scene, private shared query-plan dictionary, paired decoder, native adapter and maintained runner. Clarify three internal responsibilities: scene-owned contextual records plus organization; query-owned entrance/access planning; branch-owned member competition/integration. Record identity, exact H blocks, self ledger and provenance remain attached to the prepared scene. The query plan references those records and holds w/P/legal support; full/restricted branches keep their own C and calibrated member results. These are existing data dependencies, not a new public organization framework or serialized snapshot service.

Implement and measure the following in dependency order, keeping historical H's O-statistics/H-values equation as the control. Start with the first two; only add the later cheap opportunities where profiling and code locality justify them.

1. Avoid computing A's discarded global joint likelihood for ordinary O/H execution. Dispatch the actual read law before creating an unused normalization. Historical A/B still require it; existing requested diagnostic log weights retain their old meaning. Validate real output tensors against member-score shape/device/dtype without allocating a discarded distribution merely as a reference. Execution counters must reflect the operations actually run.
2. Deduplicate work-count reductions and defer scalar CPU conversion to the public decode/reporting boundary. Accumulate detached device count tensors across tiles, convert once, preserve exact integer dictionaries, shared-work charging and existing checkpoint behavior. Logical forward-work dictionaries and separately profiled checkpoint-replay work remain distinct; do not double-count replay or pretend the old forward counters included it. Preserve actual replay profiling without repeating reporting transfers inside backward. Do not silently disable work measurement or remove genuine finite/shape/ownership checks. Compare old/new count values, allowing only documented changes from arithmetic actually removed.
3. Reuse identical candidate-prior/statistic results only within the same read and autograd lifetime. Full and restricted candidates have different moments and calibrated scores. Cache static eligible counts and stable role/original-ID permutations at scene preparation; retain query-dependent score sorting and shortest-prefix decisions. Reuse the address query affine across M/E calls and member query affine across paired branches only where the same learned projections and dependencies apply.
4. Consider a full-reader scene row-mass shortcut, a packed contraction or existing selected-index reuse only after its assumptions and measured benefit are clear. Full support may be scene-static under ordinary positive proposals, but fixed proposals/address interventions require their actual support semantics. Keep the existing general path for those cases, not an approximation hidden as a cache. Restricted d_at remains query-dependent. Do not allocate a batch×query×address×member tensor or claim fewer contractions when arithmetic is merely packed into fewer kernel launches.

Reuse native fields, gradients, replay and checkpoints to establish E's FP32-equivalent function before attributing changes to U. Legacy trace fields such as joint log weights must not be silently replaced by typed/H probabilities under the same names. Diagnostics can compute a legacy quantity on demand; ordinary U prediction should not compute p^O or A just to populate unused trace scaffolding. Preserve historical classes/schemas and their mathematical behavior. A thin versioned U core with the same learned tensors can reuse a private H-contribution helper and return that one H tensor for both gate and value statistics. Do not duplicate the large historical decoder or create a new training engine.

## Stage U: change only the strength-statistic dependency

Implement U after the exact E comparison. Reuse the existing typed H operator; do not call O merely to discard its result. Trace final probabilities, actual strength-statistic probabilities, expectation, entropy, gate contrast/strength, selected original IDs and contribution. U computes statistics directly from the final numerically normalized H tensor and uses that same tensor for values; do not pass it through another conditional normalization merely to compute statistics. Existing joint_role_statistics renormalizes its input, so use a small internal normalized-probability statistics helper or an equivalent private seam for U while preserving historical behavior. Keep the final H numerical normalization itself. Preserve all applied relation blocks, their final-cycle stage labels, M identity self access and the existing EE diagonal convention.

Use the fixed TRAIN M1/M2/M5/M12 panel and historical input-selected detail queries. At original H100 and a retained later H milestone, compare historical H/E/U at identical prepared weights and contexts. Separate changes to expectation, entropy, strength, context norms, fields, complete native stencils and gradients. Verify the context-change identity above. This transplant measures the mechanism and compatibility with trained H weights; it is not a trained U endpoint and cannot by itself reject U's learnability.

Test exact empty/invalid-role behavior and unchanged singleton behavior. Check useful functional degeneracies such as one permitted organization row. Constant likelihood or accidental equality p^H=p^O can give equal forward reads without equal score Jacobians, so do not require general U/H gradient parity. Validate U's own correct native gradients instead. Preserve positive-softmax handling, score variance floor, entropy clamps and final H numerical renormalization; their removal would add an avoidable numerical experiment.

Track absolute native-loss gradients through typed strengths, value projections, address/member scorers and actual applied relation gates. Report gate contrast, derivative, context norm and downstream sensitivity together; a nonzero gate derivative alone does not show that the physical loss uses its context. A finite-dose fixed-marginal organization intervention can now change both allocation and strength. Record those effects separately; do not describe it as isolated value redistribution. Query-specific directory perturbations remain pointwise tests, never native derivatives assembled from separately modified stencil points.

## Reuse mature evidence before making a second architectural decision

At execution start, inspect the existing formal lineage once. If epoch 500 and its completion are available, perform one complete-native H500 versus retained ME500 review using the existing VAL89, small fixed field/interface panel and spatial organization style. Reuse unchanged reference arrays. Report all nine mean/p90/worst errors, per-M/local peaks and own-full→restricted penalties. Keep sampled engine validation separate. This evaluation is authorized evidence work under execution of this plan; it is not a restart or extension of R13-6.

If R13-6 is still running, do exact cleanup and bounded fixed-checkpoint diagnostics using H100 or a retained completed milestone. Do not attach a monitor, pause the old run, repeatedly poll it, or spend the task waiting. When mature evaluation is actually needed, use the then-available declared checkpoint once; compare matching ages where possible and label any remaining age difference. An unavailable H500 is not a reason to block independent code cleanup or silently label H400 as H500. Avoid duplicate complete VAL89 evaluations at several nearby milestones just because they are available.

Use the full/restricted comparison to distinguish three sources of error: contextual/full-predictor quality, approximation from admission, and changed integration/strength. Do not attribute all temperature regressions to the gate split. Likewise, high EE/EM row overlap is descriptive evidence; U does not automatically merge or diversify rows. Retain broad cooperative relationships and interpret organization as permitted multi-member access, not recovered physical causality.

## Bounded qualification, formal candidate and autonomous continuation

The plan's only changed scientific candidate is U. E is an equation-preserving control and can reuse the historical H lineages after verified replay; no new E100 or O100 baseline fit is required. At most one independent scratch TRAIN600 formal candidate is proposed: `thermal_r13_7u_single_distribution_train600_formal_v1`. Diagnose and repair U during qualification before that launch; do not expand to a portfolio or silently alter the scientific equation during a formal run. If a formal defect nevertheless necessitates a restart, preserve the failed lineage and use a distinct corrected run identity. All spent formal epochs count cumulatively against the 500-epoch round cap; a restart does not renew that allowance or automatically authorize a second full trajectory.

Use the existing input-only TRAIN150/VAL22 qualification membership. Start with only the epochs needed to verify actual preparation, decode, native backward, early learning health, loss curves, checkpointing and a clean pause/resume, normally 6–12 per changed path. The entire round's segmented allowance is 48 epochs including retries and controls, below the standing 500-epoch ceiling. Do not consume it as a target, use it to claim architectural maturity, or respond to a software defect by fitting for many more epochs.

Reuse the ordinary early-health diagnostics to detect nonfinite execution, scene-wide all-empty routing, widespread loss of previously active gradients or loss of scene/control dependence while training loss falls. Stop only the owned faulty candidate at a safe checkpoint, retain its evidence, diagnose and repair before longer training. Legitimate empty/singleton roles and tiny gradients with tiny denominators are not automatic collapse. Inspect gate contrast/strength, native-loss sensitivity and selected-member context together. Do not add a forced role winner, positive amplitude floor, delayed routing permission, entropy/diversity reward or new monitoring framework. Repair and continue the bounded investigation instead of delivering the first failed attempt as the final scientific conclusion.

Once qualified, train the independent U formal lineage on all TRAIN600 through 100 epochs. Match the original seed-0 learned tensor inventory/initial values, TRAIN-only normalization, ordered case membership, query draws, precision, physical objectives and constant schedule. Qualification weights are not promoted into it. Compare completed U100 against retained H100 and ME100; mature H500/ME500 are contextual references until U reaches comparable age. Save actual curves and full history, evaluate full-native VAL89 plus the fixed local/organization/cost panel, and retain a resumable state. Do not substitute sampled training loss or a short qualification result for this review.

After the completed formal 100-epoch review, the user authorizes Codex to decide whether U should resume in the same lineage to **500 total epochs**. Use its learning trajectory, fieldwise/local fidelity, native input dependence, organization mechanism and complete costs; document the decision. All early metrics need not win, but a concrete collapse or inconsistent equation needs repair rather than blind continuation. A 100-epoch miss is not a mature architecture verdict. If continuation is not justified, deliver the evidence, specific next correction and tested recovery commands rather than quietly declaring the direction exhausted.

The maximum cumulative new full-data fitting budget is 500 epochs: the initial 100 plus at most 400 continuation epochs for this one formal candidate. Do not extend old A/B, ME or R13-6 runs, add another formal arm, or continue beyond 500. Preserve the candidate's exact saved weights, parameter/RNG/provider/sampler state, membership, normalization, objectives and schedule. Retain latest, one best-field alias, declared 100-epoch milestones and curves every 10 epochs; no surprise response objective at 101. Test real engine 100→101 recovery under a 500-epoch horizon for the actual new workflow identity; a mocked runner call does not establish that the shared engine accepts continuation.

For the final 100→500 continuation, finish substantive testing/report review/cleanup, confirm successful resume/startup once, and end the task. **No observer remains:** no Luna monitor, polling, completion wait, near-completion check or follow-up automation. Ordinary in-process progress/curves/checkpoints continue. Report 500 completion as unobserved at handoff. If no continuation is chosen, the task also ends after the completed100 review and closeout.

## Execution, storage and reporting discipline

Use available physical GPUs for independent diagnostics or qualification when this helps, preserving unrelated workloads. At the planning snapshot all GPUs were occupied, including the owned R13-6 continuation on GPU 1; do not infer later availability or interrupt it. Resolve actual device mapping/ownership at execution time, without device-identity locks, reservations or a new admission service. Compare timings on isolated or explicitly described hardware conditions. No multi-device training framework is needed.

Every segmented and initial formal 100-epoch launch receives a read-only Luna agent with a minimal exact-run handoff: owned path, epoch stop, log/checkpoint/curve paths, one startup/epoch-time/ETA check, and a meaningful next inspection near 80–90% of estimated remaining duration or a declared early-health checkpoint. Earlier checks need a concrete failure or changed estimate. Luna notifies only completion, failure, meaningful delay or required action. The parent does useful independent work or rests; it must not duplicate polling. The final 100→500 continuation is the startup-only exception above. Luna remains available for other lightweight review tasks.

All formal/development Thermal forward aliases go under `HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/`; never directly under `Trained_Results`. Store actual payloads on verified DATA2, with the canonical alias pointing to the same lineage. Segmented and disposable diagnostics can stay under the round's ignored diagnostics root. Preserve the mandatory workspace symlinks and use existing datasets/caches in place. Before materialization, record current free capacity, explicit retained/temporary caps, estimated peak and headroom. Proposed execution caps are 16 GiB retained, 2 GiB temporary, at least 100 GiB DATA2 and 20 GiB home headroom; reduce before writing if estimates exceed them. Home receives only small durable text. Do not copy datasets, full source trees or checkpoints merely to compare read hooks.

Focused checks must establish E's fields/native observables/live-gradient/replay equivalence; historical A/B/O/H loader and requested-trace semantics; actual count equality and charged removed work; U's context identity, correct new gradients, empty/singleton behavior and unchanged upstream plan; native separate/paired execution; and actual checkpoint recovery including the new workflow's continuation path. Broaden testing only when modifications or failures justify it. Actual execution and empirical timing take precedence over preflight assertions. Do not add hashes, freezes, baseline snapshots, new blocking research gates, contracts or compatibility layers by default. Preserve existing security, data/TEST boundaries, job ownership, storage/checkpoint and upload protections.

The shared architecture remains generic: prescribed geometry/material/boundary/control inputs enter their owning objects, M/E are input roles, and one joint physical head predicts the fields. Keep two MM→EM→EE→ME cycles for this controlled round. No Thermal-specific flow/temperature split, heat-affine correction, heat-independent routing or flow constraint is introduced. No inverse fitting, new solver campaign or pristine TEST use is included. Defer candidate-independent score calibration, new cutoffs/entrance95, row merging, forced geometric locality, new organization scorers and aggressive compression to separately justified later experiments.

Write an English report explaining the positive changes first, then concrete gains/misses and next evidence separately for predictor, organizer and execution. Include a compact figure index, actual loss/validation trajectories across resumes, nine-field fidelity and local interfaces/peaks, same-scene spatial organization comparisons, fixed-q entrance→row competition→member→strength traces, meaningful finite-dose controls and complete latency/memory. In the new trace, make the one distribution shared by statistics and values visually explicit. Keep actual selected/executed member work separate from full address contractions and complete runtime. Retain existing whole-scene and broad cooperative views rather than imposing a near-neighbor story.

Use Python for selected scientific figures, PDF masters with small relative inline PNG previews, matching scales and English labels. Retain numerical evidence and reuse unchanged reference arrays. Inspect all included figures/rendered Markdown and resolve local links; paragraphs and captions stay on one continuous source line. A few useful plots beat exhaustive exports or a large count of weak diagnostics. Generated arrays, figures and one-time renderers remain ignored on DATA2.

Deliver relevant code/tests/configs, the report, canonical run links, actual launch/recovery commands and the formal continuation decision. Use the maintained training/resume/evaluation paths rather than another runner. Preserve preexisting user changes, including the currently edited R13-6 plan and runtime files. Complete proportionate round-owned cleanup, retaining scientific checkpoints, histories, selected native references, metrics/provenance and key figures. Record exact removals/bytes, retained evidence and remaining capacity. Audit the entire outgoing commit range, keep one-time artifacts out of Git, commit/push durable changes on the current non-default branch through existing hooks, and verify remote/local tips. Preparing this plan does not modify or resume the active R13-6 model.
