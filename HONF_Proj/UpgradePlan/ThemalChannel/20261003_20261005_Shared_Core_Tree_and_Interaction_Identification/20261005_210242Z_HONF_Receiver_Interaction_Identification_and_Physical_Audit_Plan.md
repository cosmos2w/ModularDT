# HONF: identify useful receiver–source interaction without rebuilding the Tree

## 0. Purpose and decision

**Research goal:** learn an organization of module/environment interactions from multi-field reconstruction, then reuse that organization in modular inverse design. The physical predictor is the learning vehicle; neither a small K nor a fast execution path alone establishes the representation.

**Reviewed basis:** `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `6c65d36be5a28f5da8f1bfe5f771c53ceebf945f`, and `docs/reports/HONF_Lean_Interaction_Reset_Development_Report.md`. Read the current branch and AGENTS.md before implementation; identify any subsequent changes rather than assuming this revision is still HEAD.

**Decision:** retain G-fast as the broad-field development reference and Tensor-H as a useful thermal-focused research result. Do not return to recursive Tree construction, add a K/diversity penalty, continue the old formal jobs, or train an inverse generator in this round.

The next scientific question is narrower and more informative than another sparsity experiment:

> With the same trained physical backbone held fixed, does an explicitly receiver–source-dependent control term improve reconstruction and previously untrained physical responses beyond equally trained source-only and receiver-only control terms?

Deliver two actual trained interface fits, not a diagnostic-only report:

- **H-add:** source-only plus receiver-only control corrections, with no explicit joint term in the query-control logits.
- **H-joint:** the same parameters and physical foundation, including the explicit joint receiver–source term.

Both reuse the once-wrapper tensor plan. Their new correction acts on **QM and QE only**, including the query reads used by P0/P1/P2. Existing MM/ME/EM, global calibration, coarse/local context, and all fine physical values remain present. The narrower query-interface scope is intentional: it isolates the receiver question without simultaneously changing every transport mechanism.

Both start from the **same frozen development G-fast e1000 backbone**. Fit the small interface for 500 development epochs after a real e100 review; a matched extension to 1000 interface-fit epochs is available within the budget. The backbone is not silently unfrozen after an unfavorable result.

Also execute the already prepared, bounded **12-attempt local physical-reference audit** after model selection, as specified in Section 8. Adoption of this plan and its prompt authorizes that specific audit, not a new training atlas, CFD campaign, or inverse search. Respect any still-binding cumulative solver allowance; if it cannot accommodate the request, report that explicit conflict and complete the neural experiment without substituting teacher outputs for reference truth.

## 1. What the last round established

Preserve these conclusions in the new report rather than rediscovering them:

| Topic | Evidence from the completed reset | Meaning |
|---|---|---|
| Tree cost | 144 index builds for eight cases; 18 per case; 10,800 implied per full 600-case epoch | The repeated procedural planner was an actual measured bottleneck, not merely an explanation based on parameter count. |
| G-fast specialization | Representative effective-update savings of 2.24% at low M and 5.11% at high M | Useful same-operator simplification, but not the main explanation of the much larger Tree-to-tensor difference. |
| Practical tensor cost | Complete development epochs: 9.78937 s G-fast, 12.39667 s Tensor-H; 26.63% overhead | The new organizer is affordable enough for a meaningful learning experiment. These are subset epochs, not a measured full-data Tensor ETA. |
| Reconstruction | Tensor-H versus G-fast: fluid/surface/material T −19.95%/−12.05%/−11.26%; pressure +21.26% | A real thermal tradeoff, not uniform superiority. All four flow-channel means worsen. |
| Admission | Exact case K = 1–4, mean 2.136 on all 22; variation within M3/M5/M7 | A learned case-dependent admission mechanism exists. The four displayed K=2 cases are not the full population. |
| Receiver-specific utility | Reference-mean receiver access changes core metrics by at most 0.003% on fixed4 | Useful receiver-specific organization was not established. |
| Branch reliance | Zero residual changes fixed4 fluid/surface/material T RMSE by +0.427%/+2.024%/+1.525% | Most of the tested thermal advantage survives in the co-adapted physical base; this is not a numerical decomposition of the all22 trained-model gap. |
| Physical work | Both arms execute 43,972,896 padded fine rows in 3,542 calls on all22 | Sparse control coefficients did not remove physical work or allocated control arithmetic. |
| Response evidence | TRAIN0348 is exposed training; local model AD/FD agrees; two cross-arm sensitivity signs differ | Numerical differentiation does not establish the correct physical response. |

Maintain important qualification details: initial port T worsened in every case while refined final port T improved in every case; material T at M10 worsened despite the overall gain; q-proxy maximum worsened despite its better mean. The report's edge-pressure difference is not interchangeable with its complete pressure-field RMSE or the atlas's 8%-band pressure functional.

The formal histories remain unchanged: Tree3501 has no saved checkpoint after its interrupted 97 completed epochs; Global3502 has a resumable e400 state. The jobs were already stopped before the reset arrived. Never relabel them as gracefully paused, infer recoverable weights from scalar logs, or reuse their full-data states in this subset experiment.

## 2. The architectural diagnosis to test

The code's current raw query control is of the form

\[
U_c(q,s)=\sum_e a_e(q)\,b_e(s)\,\gamma_{ec}.
\]

Here `c` is a small gain/score channel, `a` is receiver access, `b` is a source density, and `gamma` comes from collective donor content. The current `phase_actions` subtracts one reference mean per case/route/channel before the residual's tanh.

That removes an overall constant but does **not** isolate receiver–source interaction. In particular, if all receivers have the same access `a_e(q)=a_e^0`, then

\[
U_c(q,s)-\overline U_c
=\sum_e a_e^0\gamma_{ec}(b_e(s)-\overline b_e),
\]

which can be nonzero and source dependent while being identical at every receiver. The model can therefore obtain a useful source filter without solving the receiver-organization question. This is a limitation of the previous experimental parameterization, **not evidence that Codex implemented its grand-mean specification incorrectly**.

The receiver scorer also uses absolute-coordinate keys and group keys, with bounded dot-product logits. The explicit relative-distance prior currently appears in donor formation, not receiver-to-group access. The observed weak spatial access is consistent with receiver selection contributing little beyond the mature physical reader, but the code alone does not prove why training chose this solution.

Finally, jointly training the complete physical model allows an added branch to change the optimization trajectory and leave its gains in the ordinary base. That can be useful for prediction, but it prevents attributing the trained-model difference directly to a reusable interface at inference.

The experiment below addresses these issues without forcing any interaction to be nonzero.

## 3. Source map and implementation boundary

Review these maintained files, then make the smallest opt-in changes:

| Existing path, relative to HONF_Proj | Relevant responsibility |
|---|---|
| `src/honf_forward_core/interface_fields/tensor_source_group_residual.py` | Once-wrapper plan, admission/donor projections, receiver access, phase controls, grand-mean centering, and residual reductions. |
| `src/honf_forward_core/interface_fields/typed_hypergraph_field.py` | Native fine values, source eligibility, projected controls, G-fast reduction and QE attention. |
| `src/honf_forward_core/interface_fields/global_control_hypergraph.py` | Base full-access case controls and exact specialization. |
| `src/honf_forward_core/interface_fields/core.py` | Opt-in construction and phase-shared plan plumbing. |
| `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` | Physical P0/P1/P2 wrapper, predicted ports and Stage-A. |
| `Case_ThermalChannel/src/channelthermal/training/lean_attachment.py` | Existing e500-only attachment contract. Do not weaken it to accept an unrelated e1000 experiment. |
| `Case_ThermalChannel/src/channelthermal/training/epoch.py` | Real losses, accumulation, gradients and measured work. |
| `Case_ThermalChannel/src/channelthermal/workflows/train_forward.py` | Epoch boundaries, selection, save and resume behavior. |
| `Case_ThermalChannel/src/channelthermal/interaction_evidence/reference_adapter.py` | Exact-input local reference replay; no teacher fallback. |
| `src/honf_forward_core/evaluation/organization_statistics.py` and `typed_work_evidence.py` | Reuse actual plan/action/work export rather than building a second inferred graph. |

A small new `tensor_query_interaction.py` module is preferable to modifying the old Tensor-H semantics in place. A new maintained interface-fit entry/profile may own new optimizer state and the explicit frozen-backbone lineage. Reuse the native trainer, datasets, trusted loading, losses and report utilities. Do not copy an entire workflow merely to change one option.

Keep all historical readers, defaults, checkpoints, failed numerical comparisons and replay paths. No custom CUDA/Triton kernel project or broad refactor is part of this experiment.

## 4. Mathematical design

### 4.1 Case-level source plan

Retain the current proposal semantics: one shared-function proposal per present module and one environmental-background proposal. Proposal capacity and physical valid proposal count are not K. No slot-ID embedding is introduced.

Use one input-only plan per complete wrapper, carried explicitly through P0/P1/P2. Inputs include the current design/context and initial encoded source states; no target fields, errors, hidden design answers or inverse objective enter planning. Phase content is refreshed from live physical states, but the membership plan is not rebuilt at each phase.

Keep measure-normalized donor densities for M and E separately:

\[
\sum_{s\in t}\mu_s^t b_{es}^t=1,\qquad
\mu_s^t\ge0,\qquad \sum_{s\in t}\mu_s^t=1.
\]

Empty source types produce zero contributions and no false normalization claim. Invalid/padded sources are always excluded. Admission remains a learned masked simplex distribution, with exact K counted only at the exact sparse stage. There is no K, entropy, uniform-use or diversity objective.

Reuse the previous exposure schedule, **indexed by new interface-fit age**, not the frozen backbone's age:

- Fit epochs 1–100: soft admission and measure-softmax donors.
- Fit epochs 101–200: the declared linear blend.
- Fit epochs 201 onward: exact sparse projections.

Both arms train and evaluate the same stage-dependent function. Do not reintroduce a hard/soft shadow. A zero coefficient is not guaranteed to receive a restorative gradient; inspect actual learning rather than claiming that sparse projection guarantees recovery.

### 4.2 Receiver access with a common, weak geometric initialization

Both arms use the same receiver-access module and the same initial parameters. Supply an explicit signed relative receiver/anchor geometry feature, in the native normalized coordinate frame. Use a weak smooth distance bias to give the initially zero output head a nonconstant interaction basis:

\[
\ell_e(q)=s_\theta(q,e,c)
-\alpha\log\left(1+\|\widetilde q-\widetilde c_e\|^2/\ell_0^2\right),
\]

\[
a_e(q)=\frac{\pi_e\exp(\ell_e(q))}
{\sum_f\pi_f\exp(\ell_f(q))}.
\]

Defaults: `ell_0=0.25` in adapter-normalized coordinates and `alpha=0.25`. The background proposal may use zero distance bias to represent a broad context. Record that explicit background convention. Use stable exponentiation and exact masked eligibility. This is an initialization/modeling prior, not a measured physical interaction law or a hard locality rule. The learned score can overcome it.

The current key formulation may be retained with the relative feature/bias added; a small shared residual score is sufficient. Do not replace it with a large query-by-source MLP. Apply identical access architecture, prior and computation to H-add and H-joint.

Do not claim independently sparse query subgraphs: this access is positive to every admitted group unless its admission is zero. Kq can still equal case K. The question here is whether the **different weights** produce useful receiver-dependent actions.

### 4.3 Fixed, input-only receiver reference measure

For QM/QE, compute reference means over the **complete positive-weight input receiver-anchor panel**, not just its eight role centroids. This remains small-channel access work: it does not run an additional physical decoder on the reference panel. Reuse the same panel and normalized measure for both arms and all phases of a wrapper. It is independent of the random query minibatch and of requested query chunking.

If the adapter supplies no such panel, use its explicit positive-measure environmental catalogue as the documented reference. Do not silently substitute the live query batch. Retain original anchor IDs, roles, measures and coordinate joins in the export.

This changes the centering reference compared with old Tensor-H, so the new candidate is not an exact conversion of its e1000 weights. The new fits attach at zero residual to the same G-fast backbone instead.

Write the reference receiver measure as `nu`, normalized on its positive support. The following factorized identities apply to the Cartesian product of eligible query receivers and a source type. **They are not a projection theorem for arbitrary masks such as MM self-exclusion.** This is one reason to restrict the new joint test to QM/QE.

### 4.4 Separate global, source-only, receiver-only and joint control content

For one route and small control channel, define

\[
\bar a_e=\sum_r\nu_r a_e(q_r),\qquad
\bar b_e=\sum_s\mu_s b_e(s).
\]

For an ordinary nonempty normalized donor row, `bar b_e=1`, but use the measured value in code to handle numerical arithmetic and empty types correctly.

Define

\[
C_c=\sum_e\bar a_e\bar b_e\gamma_{ec},
\]

\[
S_c(s)=\sum_e\bar a_e[b_e(s)-\bar b_e]\gamma_{ec},
\]

\[
R_c(q)=\sum_e[a_e(q)-\bar a_e]\bar b_e\gamma_{ec},
\]

\[
I_c(q,s)=\sum_e[a_e(q)-\bar a_e]
[b_e(s)-\bar b_e]\gamma_{ec}.
\]

Then, before nonlinear gain conversion,

\[
U=C+S+R+I.
\]

`S` can reweight a particular donor the same way everywhere. `R` can recalibrate a receiver the same way for every donor. `I` is the explicit joint contrast: which donor weighting changes with the receiver.

The key identities are

\[
\sum_r\nu_r I(q_r,s)=0,\qquad
\sum_s\mu_s I(q,s)=0.
\]

Constant receiver access or source-independent donor densities makes I zero. **Zero is an admissible outcome.** These identities do not reward a colorful map, impose a target K, or force an improvement.

Use the same group content and gamma parameters to form S/R/I in both arms. This gives identical trainable architecture and parameter count, rather than adding a larger joint-control network only to the candidate.

### 4.5 Two trained operators, one explicit difference

Keep the inherited G-fast case-global affine action `a_base` frozen. Preserve the existing small bounded residual convention and bias/order of all native reductions:

\[
a_{\mathrm{add}}(q,s)
=a_{\mathrm{base}}+\tanh[S(s)+R(q)],
\]

\[
a_{\mathrm{joint}}(q,s)
=a_{\mathrm{base}}+\tanh[S(s)+R(q)+I(q,s)].
\]

The native gain remains its existing `1+tanh(a)`. QE retains the separate score and head-gain channel conventions; its case-constant base score cancels and stays omitted as in the release reader. Preserve softmax normalization over unique physical sources and native source measures.

The double nonlinearity follows the current residual interface rather than silently changing the base scale. Do not assert that post-tanh means are exactly zero. S and R can themselves interact through the nonlinear gain and the physical reader. Thus H-add is a **separable preactivation-control competitor**, not a physically additive simulator or a model without all interactions.

H-joint versus H-add specifically tests the extra jointly varying control term. It does not prove uniqueness against every possible nongraph architecture, nor identify fundamental many-body causal laws.

Do not add I to MM/ME/EM in this round. Those physical transport paths remain exactly the base paths; P0/P1 query changes may still affect later states through predicted ports and Stage-A, which must remain visible in the dependency description.

### 4.6 Collective content and provenance

Use the existing per-source nonlinear content encoders before measure-weighted pooling. Group summaries construct the small control content, while all original fine source values remain distinct through the native nonlinear physical messages.

Both M and E control donor types remain explicit. Collective content may be nonlinear in the pooled donor information and may use the phase and prescribed context, as already supported. Planning, base calibration, reference centering and coarse/local context are separate full-information paths.

Subtracting a reference mean can give a signed contrast on a source with zero direct group membership. Do not draw that as a newly admitted control donor. Distinguish:

- nonnegative membership/donor measure;
- direct fixed-plan content dependence;
- signed action contrast and its reference compensation;
- complete physical dependence through all other paths.

### 4.7 Initialization and gradient discipline

Freeze every inherited G-fast parameter and frozen Stage-A parameter. Initialize the new final gamma heads to zero, but keep nonuniform input-based donor/access bases and nonzero intermediate embeddings. Verify that gamma receives a real nonzero task gradient on the first native update, and admission/donor/access/content parameters receive gradients subsequently.

The H-add and H-joint models must match G-fast at attachment within the current native output tolerances, with zero new correction. There is no reason to reset or retrain the inherited physical model to run this test.

Freezing parameters is **not** permission to wrap the physical forward or Stage-A in `no_grad`: gradients from the task still have to propagate through fixed physical layers to the new controls. Keep the required activations/recomputation. Conversely, do not allocate or step optimizer state for frozen parameters.

Retain input gradients during evaluation. Do not cache hidden states across different designs or heat changes simply because parameters are frozen.

After training, removing all new corrections must return the unchanged G-fast function. This is the direct check that a gain has not been absorbed by the physical backbone. It does not, by itself, identify the joint term; use the matched trained H-add comparison and I-only intervention for that.

## 5. Implementation and engineering work

### 5.1 Bounded saved-weight audit before fitting

Spend at most 30 minutes and 40 ordinary native wrappers here. On the four fixed representatives, decompose the old Tensor-H QM/QE pre-tanh action using its **original** reference panel and verify exact algebraic reconstruction within numeric tolerances. Report the source-only, receiver-only and joint contributions on both the declared reference measure and full-grid receiver samples, keeping those measures distinct.

On at most two of those cases, intervene on the joint component only while keeping the current old source/receiver terms, weights, fine values and other routes intact. This localizes the old model's behavior; it does not decide the new model's coefficients or supply training targets.

Inspect actual old receiver-logit ranges and access spatial variance. Do not infer that bounded logits mathematically force constancy, or that one geometry feature guarantees a useful group.

### 5.2 Tensor implementation

Use batched matrix products over groups and the small control channels. Cache the reference access means once in each wrapper-owned plan. Phase gamma/content changes do not require a new reference-coordinate access computation when plan keys/access are unchanged.

The group factorization of I can be evaluated directly from centered A and B; there is no need to form an additional Q-by-source-by-hidden-dimension tensor. Preserve query chunking and evaluate the fine physical values only once per required native read.

The base fast sum plus small correction reduction remains appropriate. Benchmark actual complete update/epoch costs rather than claiming a saving from coefficient zeros. No custom packed executor, control compaction, CPU source sorting, or recursive per-case planner is introduced.

### 5.3 Compatibility and numerical tests

Reuse the existing tests and add focused cases for:

- weighted U=C+S+R+I reconstruction;
- both weighted-zero identities on the declared reference product measure;
- constant receiver access and uniform donor density producing zero I;
- unequal-mass duplicate environmental atoms and correct pulled-back gradients;
- source/module permutation and query order/chunk invariance;
- empty source types, invalid padding and M1;
- identical zero attachment and first-step gamma gradient;
- unchanged frozen physical weights, buffers and normalizers after actual fitting steps;
- I-only removal versus removal of every new correction;
- one shared plan across P0/P1/P2, and fresh plans for changed designs;
- actual release-reader versus saved training-reader outputs and input gradients.

Native checks use low/high M, complete predicted-port coupling, real source measures, and actual optimizer steps. Constructed tensor checks supplement, not replace, them.

Keep the original generic-Global versus G-fast strict VJP misses visible. This new experiment compares both arms through the same opt-in G-fast execution; it does not automatically certify that earlier strict equivalence. Allow only one bounded numerical investigation, no tolerance widening or wholesale FP64 conversion to manufacture a pass. Report used/unused/failed gradient paths separately.

### 5.4 Minimal checkpoint durability improvement

There was a concrete failure: the previous formal Tree lost all 97 trained epochs because its first scheduled save was e100 and interruption did not save a resumable state. Keep scientific selection/milestone cadence at100, but add a supported **stop request** that saves one normal latest checkpoint at the next completed epoch boundary and exits.

Use the existing trainer/checkpoint writer, an ordinary request flag and atomic replacement. A signal handler may set the flag only; it must not serialize in the middle of a GPU call or partially accumulated optimizer update. Existing RNG, normalizer, sampler and campaign state must be saved. Do not invent a new checkpoint service or cryptographic framework.

Test with a tiny disposable trainer: request stop, resume, and compare the next update to uninterrupted execution. Do not signal the historical jobs or claim retroactive recovery. Bound this engineering task to30 minutes; an absent durable implementation must be stated, not mocked by an empty test. It does not justify delaying the scientific fits for hours.

## 6. Training protocol and budgets

### 6.1 Dataset and lineage

- Use the unchanged `fixed25_v1` manifest: 150 training cases and22 exposed validation cases; all available training module-count categories retained.
- Use the development Run3601 G-fast **selected/exact e1000** as the common physical source. No formal3502 weights, full-data normalization, or Tensor-H-derived physical weights enter the pair.
- Retain selected-training normalizers and the checkpoint-owned Stage-A transform. Do not refit from the 22 validation cases or from new reference outcomes.
- Both arms use identical new-interface initialization, case/query streams, seeds, native losses, physical-denominator policy2, Q1024 and effective48/micro8.
- Create new fit identities. The old e500-only attachment checks remain intact. A small explicit frozen-backbone loader should record inherited e1000, new fit age, trainable names and the actual constant learning rate.
- New interface AdamW state begins empty in both arms, LR3e-4 and weight decay1e-5. No LR scheduler is inherited because the inspected native training has none. Do not retain misleading old `matched_fresh_initialization` or parent-null metadata as the description of this fit.
- The backbone has1000 historical trained epochs; the new interface has its own0–500/1000 fit epochs. Do not advertise500 frozen-backbone fitting epochs as500 additional physical-backbone updates.

The existing allowed TRAIN0348 callback may remain on the same native cadence in both arms. It is exposed training evidence and not part of the held audit. No extra null penalty, pseudo-response distillation, validation-target fitting, K penalty or loss-weight sweep is introduced.

### 6.2 Actual training stages

1. Execute real native initialization/update checks, then five complete epochs per arm to measure throughput and forecast e500/e1000 costs.
2. Reach100 fit epochs in each healthy arm; perform all22 statistics and a small actual component/gradient review.
3. Continue the matched pair to500 if the implemented experiment is sound and fitting remains meaningful. An immature result that has not beaten Dense is not itself a reason to stop at100.
4. An extension of **both arms to1000 fit epochs** is allowed only when the e400/e500 training/development trend leaves a substantive underfitting question and the measured forecast fits the remaining budget. Make this choice before viewing the new physical-reference outcomes.
5. Do not unfreeze the backbone, change the contrast definition, switch source checkpoints, or add a third scientific arm to rescue a weak endpoint. Record a negative result.

Selection uses the common saved100 field-MSE rule. Report exact matched endpoints separately if best ages differ. Both parent-only and old Tensor-H statistics are references, not replacements for the new trained comparison.

### 6.3 Limited adaptive problem solving

Codex may make at most two bounded implementation remedies based on an observed concrete failure, such as a broadcasting bug, a detached new head, a dead zero-initialization product, an invalid reference measure, or an unexpected extra physical forward. Each remedy must have actual execution evidence and remain in the record.

Before e100, one small training-only learning-rate/connectivity probe of at most40 updates may diagnose a nonlearning new interface. Its weights are not selected scientific weights; any recipe change applies to both fresh fits and is documented. Do not conduct an admission-temperature/K/geometry-length sweep on the22 cases.

Small equality misses, a hypothesis test with no benefit, and missing external physical truth are different failures. Do not let a heuristic preflight replace native execution, or spend the full budget on increasingly elaborate validation of an untrained branch.

### 6.4 Resource envelope

**Ceiling:12 aggregate GPU-associated hours and8 elapsed hours**, including failed starts, tests, both training arms, numerical and physical-response model evaluations, and final report closeout. Use GPUs1/2; GPU0 and unrelated processes remain untouched. Reserve the final45 minutes for a truthful report and durable delivery, not another incomplete run.

The12 local reference attempts have a separate ceiling of20 CPU wall minutes and remain inside the8-hour elapsed envelope. Do not launch a retry beyond the attempt count. Inspect the first actual solve's cost before assuming the remaining calls are cheap.

Target complete epoch/optimizer overhead near or below1.5× the same frozen-base interface control; measure it. This is a design target, not a software blocking gate or a promised acceleration. If overhead is unexpectedly several-fold, identify and remove duplicated work before committing to500 epochs; do not wait for a week-long forecast.

Use train/checkpoint data already local. With foreign GPU occupancy, use the other authorized GPU or run sequentially and document contention; do not wait indefinitely or kill unrelated work. Scale optional detailed diagnostics before sacrificing the primary matched fits.

## 7. Evaluation that answers A and B

### 7.1 Predictor statistics

At e100 and the selected/final fit endpoint, evaluate all22 cases with the native24-role evaluator. Retain all8 core means, paired wins, per-M strata, p90/worst cases, pressure-field versus pressure-functional distinctions, and initial versus refined ports. The four detailed cases remain0277/0291/0294/0687.

Do not hide pressure/velocity degradation behind better temperature or a single normalized MSE. This is a reconstruction-learning vehicle, so a useful interface must be assessed against the broad physical task. There is no claim of a formally noninferior model from one exposed seed.

The frozen G-fast baseline does not need another full evaluation when its checkpoint/protocol matches saved evidence; a small identity replay is sufficient to establish correct loading. Do not rerun mature1804 or the entire historical model zoo.

### 7.2 Trained and fixed-weight comparisons

Primary trained comparison: H-joint versus H-add at matched fit age and under the common saved selector.

Primary fixed-weight interventions on the four representatives:

1. H-joint with I set to zero; retain its learned S/R, group plan and physical base.
2. H-joint with all new corrections zero; recover the frozen G-fast operator.
3. One actual admitted group removed from I only, selected by input admission/participation before reading errors; S/R and other groups remain live.

Do not delete a group and then renormalize remaining admissions without naming that distinct intervention. Reuse unchanged normal arrays. Prefer12 additional wrappers to another hundreds-call ablation matrix.

A gain versus G-fast with no gain versus H-add is evidence for ordinary calibration, not the needed explicit joint interaction. A gain versus H-add with a negligible I intervention requires inspection of compensation in S/R or a training-path effect within the new heads; freezing the physical base alone does not eliminate compensation among new parameters. The matched trained contrast and frozen interventions are complementary, not interchangeable proofs.

### 7.3 Organization quality, not just K

Report allocated proposals, valid proposals, exact admitted K, donor support and actual physical work separately. Show all22 K and within-M variation. K=1 or an inactive I is a permissible negative outcome, not a reason to force more groups.

Export actual S, R, I, pre/post-tanh executed action and group membership on physical coordinates. Show the source/reference measure used. Signed contrasts are not donor probabilities. Include the measured joint-component norm and the fixed-weight field change from removing it, not just gamma norms.

A nonzero action can be physically ineffective through the downstream reader. Conversely, a nonzero field effect can worsen physical truth. Display both effect and error difference.

The fine source inventory remains dense, and no physical executor saving is expected from this test. Measure complete inference, representative input VJP, full optimizer/epoch time and allocated/reserved memory. Never present lower coefficient count as latency improvement.

### 7.4 Reuse-oriented checks without training an inverse model

Use the existing heat and module-coordinate AD/two-step FD probes on the four cases. Count support changes and distinguish same-support checks from crossings. Measure exact unchanged-heat flow targets as absolute/scaled increments, not relative-to-zero error.

No inverse generator, sampler trajectory, multi-start design search or graph-block optimizer is required. The new physical audit below is more informative than another self-consistent surrogate-only search.

## 8. Execute the small physical-response audit

### 8.1 What is authorized and why

The previous rounds repeatedly prepared but did not execute

`/data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/follow_on_reference_request_fixed4.json`.

This plan authorizes **at most12 local analytic-wake/shared-grid attempts** for its four aligned baselines and two opposite feasible fixed-total heat transfers per baseline. Reuse the exact documented design inputs and module IDs; verify that they correspond to the intended four validation representatives. Do not silently replace the request with the newer Q14 derivative directions if they differ.

Use the maintained reference adapter and exact embedded case configuration, not the neural predictor. The local generator is not CFD, and the audit does not establish a mesh-converged or SI-certified design result.

Check the existing cumulative attempt ledger and any still-binding solver restrictions before execution. If a conflicting remaining allowance prevents12 attempts, report it and seek the required authorization rather than bypassing it. Absence of the optional local generator is an explicit unavailable audit; it must not become teacher-generated truth. Continue the independent neural experiment rather than holding the entire round indefinitely.

**2026-10-05 authorization amendment:** after the neural checkpoint selection and matched500 stopping decision were sealed, the user explicitly raised the cumulative local benchmark ceiling from320 to326 to finish this exact audit. The reconciled pre-audit count is314, leaving twelve counted attempts under the amended ceiling. Failed attempts still count; no extra retry, CFD solve, training-atlas expansion, inverse search, formal restart or model reselection is authorized. The original12 aggregate GPU-hour,8 elapsed-hour and20 CPU-wall-minute audit limits remain in force across the completed neural work and this continuation.

### 8.2 Timing of outcomes and separation from fitting

Select the neural checkpoints and decide any e1000 extension **before opening these new response outcomes**. No physical audit result chooses architecture, loss weights, geometry scale, stopping checkpoint, or training examples in this round. The reference records are evaluation only; do not append them to TRAIN0348 or the150 training cases.

The correct evidence label is:

> New heat-perturbation responses not used in fitting, on four repeatedly exposed validation layouts, within the fixed-grid benchmark.

It is not an untouched layout test, an independent broad population sample, or a demonstrated inverse design.

### 8.3 Preserve alignment and numerical limitations

Generate each trial and its exact-input baseline with consistent rounding and solver settings. Compare finite responses against that generated baseline, not a nearly matching differently rounded stored state. Join fluid queries/masks, physical module IDs, surface angles and material coordinates using the maintained conventions.

For each attempt, retain status, raw output and wall time. Failed/nonconverged states are unavailable, not zero responses. The12-attempt count includes failures; do not add an undeclared mesh or step-size sweep.

Reduce reference differences in wider arithmetic from the stored output precision. Record baseline replay discrepancy, final solver convergence indicators and output-quantization scale. Do not claim these alone bound grid error. For small effects, label sign unresolved when comparable to those indicators. No positive physical interpretation follows from a tiny noisy ratio.

### 8.4 What to compare

Read the frozen G-fast, old Tensor-H, selected H-add and selected H-joint on exactly the same solved inputs. Compare:

- absolute baseline and perturbed temperature roles;
- signed and RMS finite fluid/surface/material temperature responses;
- per-module peak changes, including unchanged-own-heat receivers;
- q-normal proxy as a separately labelled proxy;
- known-null flow and the existing8%-band pressure response functional;
- zero-change predictor and H-add as distinct controls for H-joint.

Report direction/amplitude agreement where reference signals are resolved. The request's finite transfer is not automatically the same derivative as the old epsilon0.001 Q14 probe. The previously disagreeing module-position sign remains unvalidated because these12 calls change heat, not geometry.

This audit is deliberately small. A positive result supports local response transfer on these named neighborhoods, not a reliable global inverse system. A negative result tells us whether the remaining barrier is physical response error rather than differentiability or model execution.

## 9. Interpretation and next decision

Use research judgment with the actual evidence; do not install another automatic promotion/fallback framework.

| Outcome | Defensible interpretation | Next action |
|---|---|---|
| H-joint beats trained H-add; I removal worsens held reconstruction; new response audit is favorable | The explicit receiver–source query-control term earns local reusable value on a fixed capable backbone | Retain it and propose a separately reviewed limited co-adaptation or manual comparative formal run, with measured full-data cost. |
| Both new fits improve, but H-joint does not beat H-add or I remains ineffective | Source/receiver calibration is sufficient in this experiment; explicit joint-control value remains absent | Keep the cheaper/simpler fit. Do not increase K or add a Tree to force the desired story. |
| H-joint helps values but not physical finite responses | Reconstruction correction was learned, but inverse-relevant response is not yet reliable | Prioritize the particular failed response/data dependency, not a larger generator. |
| Both fail to improve with live useful gradients and adequate fit exposure | Frozen-host residual capacity/supervision is insufficient for this task | Report that scoped result. Do not claim that all hypergraph approaches are disproven or secretly unfreeze to rescue the experiment. |
| Useful joint term but material flow regression or excessive runtime | An organization mechanism exists with an unacceptable tradeoff | Preserve the measured mechanism; no stand-alone formal promotion. |
| Physical audit unavailable | Numerical/model evidence only | State the exact missing capability/authorization; do not repeat a physical-readiness claim. |

Formal3501/3502 remain stopped. No full-data optimizer starts automatically. A later formal recipe requires a separate explicit choice and measured complete-epoch cost; a5000-epoch number alone is not a scientific stopping criterion.

## 10. Reporting, visualization and delivery

Start the final report with at most350 words explaining what changed and what was gained/missed for **Predictor / Organizer / Inverse**, then state A/B/C in ordinary language. Separate a completed implementation, trained empirical result, numerical self-check and physical-reference observation.

Keep detailed role tables and protocol information, but use five main figure groups:

1. **Learning and cost:** H-add/H-joint curves against fit epochs and measured training time; inherited frozen backbone marked separately; actual work/memory.
2. **Physical fields:** stored truth, both fitted predictions and signed residuals for the four unchanged representatives, including difficult cases and flow tradeoffs.
3. **What the control represents:** source-only S, receiver-only R, joint I and actual effective gain/score, with physical donor locations and the declared reference measure; no latent norms labelled energy.
4. **Organization utility:** all22 admission K/within-M behavior and paired reconstruction; I-only removal field effects versus physical error change; show a weak/failed case as well as a favorable one.
5. **New physical response:** before/after reference temperatures, measured versus predicted finite responses and unchanged-flow leakage. If no solves completed, replace this with an explicitly unavailable panel, never a synthetic stand-in.

There are no inverse generation trails to plot because no inverse search runs. Do not manufacture them. Each figure needs a short quantitative caption and one plain-language statement of what it supports and what it does not. Retain inspected PDF masters and only necessary Markdown raster companions, under the existing ignored artifact policy.

Preserve all historical checkpoints and security measures. Do not add cryptographic contract frameworks, baseline snapshot systems, hard-coded new hash chains, production-style preflight gates or broad fallback machinery. Reuse Git history, typed configs, existing trusted loaders, ordinary tests and the maintained checkpoint writer. Existing security/artifact restrictions remain enforced.

Commit and push durable source/tests/configs/report to the existing non-default research branch after reviewing the entire outgoing object range and running the existing pre-push hook. Keep one-time profiling/rendering code, raw references, arrays, figures and scientific checkpoints local in the existing ignored locations. Verify the local/remote tip as required by AGENTS.md.

## Appendix A. Algebra evidence available with this plan

A constructed FP64 CPU test checked the proposed weighted decomposition; it did not load ModularDT checkpoints, train a model or run physical simulations. For two batches with9 reference receivers,7 sources,4 groups and3 control channels:

- `U - (C+S+R+I)` max absolute error: `4.44e-16`.
- Both weighted I means: below `1.3e-16`.
- Constant receiver access: joint max `2.38e-16`, while the old grand-mean-centered action still had RMS `0.2684`.
- Uniform donors: joint max below `8.7e-17`.
- Unequal30/70 environmental atom refinement: joint equality max `0` in that constructed example.
- A zero gamma head had zero output and nonzero task gradient norm `0.1192` for a nonconstant basis and a constructed target.

These checks support the algebra and the initialization mechanism only. Codex must perform the native implementation and learning tests above.

## Appendix B. Reading references

Source-derived facts in this plan come from the completed reset report at `6c65d36`, particularly its formal-history/cost, scientific-change, final-organizer and response sections, and the source paths in Section3. The decomposition, frozen-backbone comparison and12-attempt audit timing are **new recommendations**, not outcomes already demonstrated by that report.
