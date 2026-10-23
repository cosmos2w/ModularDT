# Stage 6 Goal Mode Plan — Fixed Softmax + Role-Consistent Query Routing + Additive Decoder

## 0. Scope and authoritative source

Work only in:

- repository: `cosmos2w/ModularDT`
- branch: `agent/honf-adaptive-correctness`
- authoritative branch head at plan preparation: `9a277b1ff06fa9f6e0f7aa54178db5a52751e266`

At task start, verify the actual checked-out branch and `HEAD`. If the branch has advanced, record the new SHA and inspect the relevant diffs before editing. Do not use `main` as the implementation reference.

This Stage 6 round intentionally **drops exchangeable HONF from the critical development path**. Do not modify or remove the exchangeable implementation; keep it as a research/ablation branch. The practical forward model now develops from the fixed-softmax modern model represented by Run 1302.

The scientific references are:

- **Run 1000** — accuracy and organizer/routing structural reference.
- **Run 1302** — practical modern fixed-softmax/additive accuracy–efficiency baseline.
- **Run 1301** — evidence that the modern additive decoder can support strong query and edge-field differentiation, but not a model to continue developing in this round.

The Stage-6 goal is:

> Build a fixed-softmax modern forward model whose hyperedges remain physically meaningful and distinguishable from environment organization through query routing and additive edge fields, while preserving Run-1302-class computational efficiency and moving predictive accuracy toward Run 1000.

The target concept is:

\[
oxed{	ext{Fixed Softmax + Role-Consistent Query Routing + Additive Decoder}}
\]

No inverse model should be implemented in this task. Stage 6 must, however, preserve a clean mechanism-level interface for the next inverse-design stage.

---

# 1. Established evidence — do not spend GPU time re-proving it

## 1.1 Run 1302 is the practical modern base

At 2500 epochs, the modern fixed organizer is slightly more accurate than the exchangeable Run 1301, materially faster in the full forward, smaller, and much better organized in environment geometry.

Key complete-split results:

- Run 1302 best pooled normalized fluid MSE: about `0.001786`.
- Run 1301 best pooled normalized fluid MSE: about `0.001800`.
- Run 1000 best pooled normalized fluid MSE: about `0.000731`.

Run 1302 full-forward median: about `28.69 ms`.
Run 1301 full-forward median: about `40.25 ms`.
Run 1000 full-forward median: about `25.29 ms`.

Run 1302 total parameters: about `3.979 M`.
Run 1301 total parameters: about `4.831 M`.
Run 1000 total parameters: about `3.509 M`.

Therefore Stage 6 must not add a heavy organizer or expensive new dense attention path.

## 1.2 Run 1302's environment organization is useful but not yet Run-1000 quality

Complete-split medians:

| Metric | Run 1000 | Run 1302 |
|---|---:|---:|
| environment edge-profile cosine | ~0.094 | ~0.260 |
| environment effective rank | ~5.24 | ~1.48 |
| largest environment occupancy | ~0.292 | ~0.859 |
| normalized region separation | ~0.304 | ~0.247 |

Run 1302 clearly preserves useful physical geometry, but one environment owner is still too dominant and the environment matrix remains lower rank than Run 1000.

## 1.3 Run 1302 loses role distinction downstream

Complete-split medians:

| Metric | Run 1000 | Run 1302 |
|---|---:|---:|
| query edge-profile cosine | ~0.436 | ~0.790 |
| query effective rank | ~3.70 | ~1.94 |
| query entropy | ~0.763 | ~0.927 |
| query spatial std | ~0.155 | ~0.078 |
| pairwise map cosine | ~0.462 | ~0.711 |

Run 1302 additive edge-field cosine is roughly `0.73–0.79` depending on physical channel.

Thus the fixed organizer produces more useful environment geometry than is preserved by the current modern mechanism/query stack.

## 1.4 Run 1302 topology degrades during late optimization

From about epoch 1500 to 2500:

- environment effective rank: about `1.76 -> 1.48`;
- largest environment occupancy: about `0.784 -> 0.859`;
- query effective rank: about `2.17 -> 1.94`;
- query profile cosine: about `0.716 -> 0.790`;
- temperature edge-field cosine: about `0.650 -> 0.728`.

Field loss continues improving while topology worsens.

Do not use field MSE alone as evidence that the scientific representation is improving.

## 1.5 Retained-mass pruning is not useful yet for Run 1302

With the same evaluation-only rule:

```text
query retained mass = 0.98
module retained mass = 0.95
```

Run 1302 removes only about `1.1%` of query-edge routes. Its soft query distribution is too diffuse.

Stage 6 should make query roles more differentiated **before** relying on gathered execution.

---

# 2. Mathematical inverse-design review: fixed hyperedges are sufficient and may be preferable

The decision to abandon exchangeable hyperedges does **not** block the planned inverse-design stage.

Let a design be \(d\), query coordinate be \(q\), forward state be \(F_	heta(d,q)\), and requested KPI/field specification be \(c^\star\).

The fixed organizer remains design-dependent:

\[
A_{mh}(d)=
\operatorname{softmax}\!\left(
W_m\,t_m(d)+b_m

ight),
\]

and, schematically,

\[
A_{eh}(d)=
\operatorname{softmax}\!\left(
W_e\,t_e(d)+b_e+B_{
m geo}(d)

ight).
\]

The hyperedge **indices** are fixed, but their assignments, masses, centroids, scales, latent content, routing, and physical contributions remain functions of the design.

For each design the forward model produces an ordered mechanism state

\[
Z_	heta(d)
=
\left[
z_1(d),\ldots,z_K(d)

ight],
\qquad K=6,
\]

where a mechanism representation can include:

\[
z_k =
\left[
h_k,\;
s_k,\;
r_k,\;
\sigma_{s,k},\;
\sigma_{r,k},\;
m_{m,k},\;
m_{e,k},\;
p_{m,k},\;
p_{e,k}

ight].
\]

The fixed index \(k\) gives the inverse model a **stable coordinate system across cases**.

This is not a disadvantage for inverse design. It is often easier than a permutation-variable latent set.

A conditional inverse can learn

\[
p_\phi(d\mid c^\star)
\]

or a hierarchical proposal

\[
c^\star

ightarrow
z^\star

ightarrow
\hat d,
\]

and use the frozen forward model for validation or differentiable correction:

\[
\mathcal L_{
m request}
=
\left\|
\mathcal K(F_	heta(\hat d))-c^\star

ight\|^2.
\]

The inverse problem is generally many-to-one; exchangeability does not solve that multimodality. A generative inverse model is still needed regardless of whether forward hyperedge indices are fixed.

The exact additive forward model also gives mechanism-resolved sensitivities:

\[
F_	heta(d,q)
=
F_{
m bg}(d,q)
+
\sum_{k=1}^{K}F_k(d,q),
\]

so for a scalar objective \(J\),

\[
rac{\partial J}{\partial d}
=
rac{\partial J}{\partial F_{
m bg}}
rac{\partial F_{
m bg}}{\partial d}
+
\sum_k
rac{\partial J}{\partial F_k}
rac{\partial F_k}{\partial d}.
\]

That decomposition is useful for later inverse interpretation and correction.

### What is lost by abandoning exchangeability?

Mainly:

1. zero-shot change of hyperedge count \(K\);
2. a single parameterization intended to represent a variable number of mechanism slots;
3. permutation-equivariant slot semantics.

None of these is required for the first inverse-design model.

Variable module count is already a different concept from variable hyperedge count: the fixed organizer can consume variable/padded module sets through `module_present` and map them into the same six learned mechanism coordinates.

For the first formal wind-farm/geothermal models, it is acceptable to choose \(K\) per model family and freeze it.

### Stage-6 inverse-readiness requirement

Do not change or remove the following forward outputs/interfaces:

- `hyper_state`;
- `mechanism_descriptor_features`;
- `hyper_source_coords`;
- `hyper_region_coords`;
- source/region scales;
- module/environment masses and purities;
- `A_mh`;
- `A_eh`;
- query routing diagnostics;
- exact additive per-edge fields when requested;
- prepared encode/organize state.

Do not implement inverse code in Stage 6.

---

# 3. Stage-6 mathematical model

Keep the Run-1302 fixed organizer and exact additive assembly.

## 3.1 Fixed soft environment organization

Keep the current fixed learned edge channels:

\[
A_{mh}
=
\operatorname{softmax}(L_m),
\]

\[
A_{eh}
=
\operatorname{softmax}(L_e+B_{
m env,geo}).
\]

Do not add entmax, hard selection, count losses, entropy losses, diversity losses, or load-balancing losses.

## 3.2 Preserve organizer identity in the mechanism state

The current descriptor-first mechanism state is

\[
	ilde h_k
=
\operatorname{LN}
\left[
D(d_k)
+
\lambda_h C(h_k)

ight],
\]

with current

\[
\lambda_h=0.35.
\]

This may suppress useful learned hyperedge identity.

Stage 6 should test a larger content residual, with the primary candidate:

\[
oxed{\lambda_h=0.70}.
\]

Do not immediately make `0.70` the canonical default. First evaluate it using existing Run-1302 weights.

## 3.3 Role-consistent soft query routing

The existing query content score is

\[
\ell^{
m content}_{qk}
=
rac{
Q(z_q)^T K(	ilde h_k)
}{\sqrt H}
+
b^{
m learned}_{qk}.
\]

Add/enable a direct soft physical role prior from the organizer's learned region geometry:

\[
b^{
m role}_{qk}
=
-rac12\,
\lambda_q\,
\min
\left[
\sum_j
\left(
rac{q_j-r_{k,j}}{\sigma_{k,j}}

ight)^2,
R^2

ight].
\]

Then

\[
oxed{
lpha_{qk}
=
\operatorname{softmax}
\left(
\ell^{
m content}_{qk}
+
b^{
m role}_{qk}

ight)
}
\]

for training.

This is still completely soft.

No exact zeros are introduced.

The current decoder already implements the required region-based query-locality pathway. Stage 6 should use that path rather than creating another query router.

The role prior is \(O(BQK)\), so it should add negligible cost relative to the existing decoder.

### Configuration clarity

If useful, add one backward-compatible field:

```json
"query_locality_strength": null
```

with semantics:

- `null`: inherit the historical `environment_locality_strength`;
- positive float: use an independent query-role strength.

Only add this field if it materially simplifies the Stage-6 profile. Missing/null must preserve existing arithmetic.

Do not create a new query-routing network.

## 3.4 Exact additive field assembly

Keep

\[
F(q)
=
F_{
m bg}(q)
+
\sum_{k=1}^{6}
lpha_{qk}F_k(q).
\]

Keep dense query-attention background.

Keep the shared edge head.

Keep pairwise routing.

Do not add a joint correction branch in the initial Stage-6 revision.

A light joint correction is a **future contingency only** if Stage 6 succeeds structurally but still fails the accuracy gate.

---

# 4. Stage 6 execution strategy: use existing checkpoints first

Do not start with training.

## 4.1 Frozen-weight screen

Use at least:

- Run 1302 epoch-1500 milestone;
- Run 1302 best-field checkpoint;
- optionally Run 1302 latest if useful for confirming late drift.

Evaluate these four configurations:

### S0 — reference

```text
mechanism_latent_residual_scale = 0.35
query_locality_mode = none
```

### S1 — richer organizer content

```text
mechanism_latent_residual_scale = 0.70
query_locality_mode = none
```

### S2 — role-consistent query routing

```text
mechanism_latent_residual_scale = 0.35
query_locality_mode = gaussian_bounded
query locality strength = 0.25
```

### S3 — combined candidate

Run only if S1 and/or S2 gives useful evidence:

```text
mechanism_latent_residual_scale = 0.70
query_locality_mode = gaussian_bounded
query locality strength = 0.25
```

No weight updates.

Use a deterministic 20-case subset first.

Run the full 90-case evaluation only for S0 and the best candidate.

Do not make a large grid search.

## 4.2 Frozen-screen metrics

Primary:

- pooled fluid MSE;
- per-channel MSE;
- case p95 MSE;
- query edge-profile cosine;
- query effective rank;
- query entropy;
- query spatial standard deviation;
- pairwise map cosine;
- per-channel additive edge-field cosine/rank;
- retained-mass query route reduction;
- retained-mass MSE change;
- decoder timing.

Environment organization is checkpoint-owned and should not change in a frozen decoder-only screen. Record it for context but do not interpret unchanged `A_eh` as a failure of the screen.

### Frozen-screen success signal

A candidate is worth training if it moves query routing materially toward Run 1000 / Run 1301-like diversity without destroying the field.

Suggested screen targets:

\[
	ext{query profile cosine}<0.65,
\]

\[
	ext{query effective rank}>2.5,
\]

\[
	ext{query entropy}<0.88,
\]

and preferably:

\[
	ext{query-route pruning}>10\%-15\%
\]

under retained mass `0.98`.

A frozen checkpoint field-MSE increase up to about `5%` is acceptable for screening because the weights were not trained under the new routing rule.

Reject a variant if field error deteriorates catastrophically or routing collapses to a single global owner.

---

# 5. Stage-6 code work

Keep implementation minimal.

## Mandatory

1. Preserve all Stage-5 diagnostics and retained-mass evaluation tools.
2. Add/use a clean Stage-6 fixed-softmax experiment profile.
3. If needed, add independent `query_locality_strength` with exact backward compatibility.
4. Ensure mechanism latent residual scale can be overridden cleanly by experiment overlay.
5. Keep dense training execution.
6. Keep retained-mass gathered execution evaluation-only/deployment-only.
7. Preserve exact additive closure.
8. Preserve Run-1302 state-dict compatibility for evaluation-only configuration overrides.

## Strongly recommended diagnostic addition

Selected/viable edge count is no longer a scientific topology metric.

Add a **small number of cheap scalar role-quality diagnostics** to normal evaluation/training output if they can be computed without material overhead.

Priority order:

1. environment edge-profile cosine;
2. normalized region-center separation;
3. query edge-profile cosine.

Do not add expensive full routing maps to every training batch.

If a query-profile cosine can be computed from the already-available `hyper_attention` inside the decoder without retaining the full map beyond the forward call, expose only the scalar.

If adding these scalars measurably slows training, keep them in milestone/full-split evaluation only.

---

# 6. One Stage-6 training profile only

After the frozen screen, prepare **one** main Stage-6 profile using the best evidence.

Suggested name:

```text
stage6_fixed_role_consistent_additive.json
```

Expected design:

```json
{
  "organizer_mode": "fixed_projection",
  "num_hyperedges": 6,

  "module_assignment_normalizer": "softmax",
  "environment_assignment_normalizer": "softmax",
  "query_assignment_normalizer": "softmax",

  "edge_selection_mode": "all",

  "mechanism_state_mode": "descriptor_first",
  "mechanism_latent_residual_scale": 0.70,

  "query_locality_mode": "gaussian_bounded",

  "field_assembly_mode": "edge_additive",
  "additive_background_mode": "dense_query_attention",

  "routing_execution": "dense"
}
```

Use the frozen screen to decide whether `0.70` and query locality are both retained. Do not force both if only one is supported.

Training:

```text
prediction LR = 3e-4
organizer LR  = 1e-4
```

Keep every other Run-1302 setting unchanged.

Do not add a new optimizer scheduler in the initial Stage-6 run.

---

# 7. Formal run policy — Codex does not launch the long run

Codex should prepare the profile, dry-run it, and provide the command.

The user controls formal training.

Recommended first budget:

```text
1500 epochs
```

Evaluate at:

```text
250
500
1000
1500
```

Only resume the same run to:

```text
2500 epochs
```

if:

- field accuracy is still improving;
- environment organization has not materially degraded;
- query/mechanism roles remain scientifically meaningful.

Do not automatically launch multiple Stage-6 formal variants.

Do not launch a 5000- or 10000-epoch run in Goal Mode.

---

# 8. Stage-6 acceptance gates

## 8.1 Accuracy

Run 1302 is the immediate baseline.

At 1500 epochs:

- field curve should be no worse than about `10%` above Run 1302 at the same epoch;
- preferably match or improve Run 1302;
- no systematic channel collapse.

At 2500 epochs if resumed:

Primary target:

\[
	ext{pooled fluid MSE}<1.6	imes 10^{-3}.
\]

Minimum acceptable:

- improve on Run 1302 best `~1.786e-3`, or
- achieve clearly better topology with no more than about `5%` accuracy penalty.

Run 1000 remains the long-term target:

\[
7.31	imes10^{-4}.
\]

Stage 6 does not have to completely close that gap in one round, but it must reduce it or provide a clearly stronger mechanism representation at comparable accuracy.

## 8.2 Environment organization

Do not accept a model whose environment organization becomes worse than Run 1302 simply to improve query plots.

Preferred:

\[
	ext{environment profile cosine}\le 0.25-0.30,
\]

\[
	ext{normalized region separation}\ge0.22,
\]

and improve largest environment occupancy from the Run-1302 value of about `0.859`.

A strong Stage-6 result should move environment effective rank upward from `~1.48`, ideally toward `>2`.

Run 1000 remains the structural target, not a mandatory one-round gate.

## 8.3 Query roles

Target:

\[
	ext{query profile cosine}<0.60,
\]

\[
	ext{query effective rank}>2.8,
\]

\[
	ext{query entropy}<0.85,
\]

with clear spatial variation.

Do not use dominant-edge occupancy alone.

## 8.4 Additive mechanisms

Per-channel edge-field profile cosine should move substantially below Run 1302's `~0.73–0.79`.

Initial target:

\[
	ext{edge-field cosine}<0.60
\]

for most physical channels.

Do not require one edge per physical variable.

Mechanisms may be multi-variable physical processes.

## 8.5 Efficiency

The new role-consistent fixed model should remain close to Run 1302 dense cost.

Gate:

- total parameter increase <= about `5%`;
- full-forward latency <= about `1.10 x` Run 1302;
- no new \(Q	imes E\) dense branch;
- no exchangeable iterative organizer.

For deployment/evaluation retained-mass pruning:

Target:

\[
	ext{query-edge route reduction}\ge15\%-20\%
\]

with:

\[
	ext{aggregate MSE increase}\le2\%,
\]

and no channel MSE increase above about `5%`.

Module-incidence pruning is **not a Stage-6 success requirement**. The Stage-5 evidence shows it is currently negligible. Do not distort module organization just to obtain a pruning percentage.

---

# 9. Late topology drift

Run 1302 became more accurate while becoming less structured from epoch 1500 to 2500.

For the Stage-6 formal run, compare each milestone's:

- environment profile cosine;
- region separation;
- query profile cosine;
- query effective rank;
- edge-field cosine;
- field MSE.

If topology clearly improves early and then deteriorates late while field MSE continues improving, record that as a separate optimization issue.

Do **not** immediately add another arbitrary representation schedule.

A later small optimizer-only intervention, such as reducing/fixing the organizer LR after role formation, may then be justified in a separate experiment.

Do not implement that preemptively in this round.

---

# 10. Contingency for the remaining Run-1000 accuracy gap

Do not implement this unless the main Stage-6 role-consistent model succeeds structurally but remains clearly accuracy-limited.

The next candidate would be a **light joint correction**:

\[
F(q)
=
F_{
m bg}(q)
+
\sum_k F_k(q)
+
\epsilon F_{
m joint}(q),
\]

where:

- \(F_{
m joint}\) is low capacity;
- it is separately exported/diagnosed;
- its gate starts small;
- it does not replace the additive mechanism fields;
- its contribution fraction is explicitly measured.

The purpose would be to recover high-order interactions that exact additive mechanisms cannot efficiently express.

This is **not part of the initial Stage-6 implementation**.

---

# 11. Inverse-readiness acceptance

Before Stage 6 closes, verify/document that a frozen fixed model exposes a stable ordered mechanism representation suitable for inverse use.

For one forward case, confirm the availability/shapes of:

```text
global_token                    [B,H]
hyper_state                     [B,K,H]
mechanism_descriptor_features   [B,K,D]
A_mh                            [B,M,K]
A_eh                            [B,E,K]
hyper_source_coords             [B,K,2]
hyper_region_coords             [B,K,2]
hyper_source_scale              [B,K,2]
hyper_region_scale              [B,K,2]
hyper_module_mass               [B,K]
hyper_env_mass                  [B,K]
pred_field_by_edge              [B,Q,K,F] when requested
```

Do not create a new learned inverse encoder.

A small documentation/helper export is acceptable if needed.

The mechanism order is fixed by the learned fixed-projection channels and therefore can be consumed directly by a future hierarchical inverse model.

---

# 12. Tests and budget limits for Codex

## Unit/integration tests

Run only relevant maintained tests after code changes.

At minimum:

- config/schema tests for any new query-locality field;
- fixed organizer regression;
- additive exact closure;
- checkpoint/state-dict compatibility;
- prepared/chunked decoding;
- retained-mass routing;
- Stage-6 profile dry-run.

## Smoke training

Only if model code changes require it.

Maximum:

```text
2 epochs
4 train batches / epoch
2 validation batches
```

Do not use smoke training to infer scientific topology.

## Frozen evaluation

This is the main intermediate test and should replace unnecessary training screens.

Use 20 cases first; full 90 cases only for the best candidate.

## Long training

Codex does not launch it.

---

# 13. Required deliverables

1. `HONF_Proj/UpgradePlan/Stage6_Fixed_Role_Consistent_GoalMode.md`
   - this plan or an updated repository copy.

2. A concise Stage-6 frozen-screen report, for example:
   `HONF_Proj/UpgradePlan/Stage6_FrozenScreen_Evaluation.md`

3. One final experiment profile:
   `stage6_fixed_role_consistent_additive.json`

4. Updated config/schema/tests only if needed.

5. Optional cheap scalar topology diagnostics if they do not harm throughput.

6. Dry-run output/provenance for the Stage-6 formal profile.

7. A final closeout report stating:
   - which frozen variant was selected;
   - effect on field error;
   - effect on query routing;
   - effect on edge-field diversity;
   - expected computational overhead;
   - exact formal launch command;
   - explicit statement that Codex did not launch a long run.

---

# 14. Explicit non-goals

Do not:

- continue exchangeable-organizer development in Stage 6;
- add entmax;
- add hard selection;
- add count/entropy/diversity/load-balancing regularization;
- add variable edge count;
- redesign the background;
- add a large new query network;
- add edge-specific decoder heads;
- change Stage-A;
- modify inverse code;
- implement joint correction before the primary Stage-6 candidate is evaluated;
- run broad hyperparameter sweeps.

---

# 15. Completion definition

Stage 6 is ready for the user-controlled formal run when:

1. the fixed Run-1302 architecture remains the base;
2. one role-consistent query-routing candidate has been selected by frozen evaluation;
3. mechanism-state content scale is set from evidence rather than guesswork;
4. exact additive closure and checkpoint compatibility remain valid;
5. dense runtime remains close to Run 1302;
6. the formal profile passes dry-run;
7. inverse-relevant mechanism outputs remain stable and documented;
8. no unnecessary exchangeable or sparse-training machinery has been reintroduced.

The central Stage-6 scientific test is:

\[
oxed{
	ext{Can the physically structured fixed organizer retain its roles through query routing and additive decoding,}
\]

while

\[
oxed{
	ext{approaching Run-1000 accuracy and becoming prune-able at execution time?}
\]

If yes, this fixed role-consistent HONF should become the forward model used to begin the inverse-design stage.

If the role-consistent model becomes structurally strong but remains accuracy-limited, the next forward change should be the isolated light joint-correction experiment—not another organizer redesign.
