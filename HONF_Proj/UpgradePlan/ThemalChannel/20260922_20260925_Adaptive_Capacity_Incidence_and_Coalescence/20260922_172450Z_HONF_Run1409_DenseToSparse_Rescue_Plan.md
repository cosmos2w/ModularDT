# Run 1409 v2 — Dense-to-Sparse Case-Budgeted HONF

## Goal

Rescue the scientific question behind Run 1409 without changing the central model family:

> Can an over-capacity hypergraph first learn a useful multi-group physical organization and only then compress itself to a case-dependent effective group count, while preserving Run-1406-like fidelity and eventually reducing real computation?

The first Run 1409 is not treated as evidence that conditional capacity is impossible. It exposed an optimization pathology:

- `Kmax=12`;
- the deterministic gate collapsed to exactly one live group for all 90 cases by epoch 50;
- validation expected optional count fell from about 10.44 at epoch 1 to 0.00135 at epoch 50;
- the one-group model still executed all valid module/environment fine pairs;
- physical accuracy was substantially worse than Run 1406.

The new experiment keeps the same scientific architecture and Run ID **1409**, but changes the optimization path so group roles are learned before they are allowed to disappear.

This is a fresh timestamped managed Run 1409. Do **not** resume the collapsed checkpoint.

---

# 1. Diagnosis of the first Run 1409

The previous model used:

\[
A^M,\; A^E,\; \alpha
\]

with a case-level hard-concrete availability plan. A closed gate was masked out of source and query entmax normalization.

That creates an important one-way optimization effect:

1. When a group gate is exactly zero in a forward pass, that group receives no source/query membership.
2. Its physical contribution is therefore zero.
3. Physical-loss gradients through that group's routing path disappear in that forward.
4. The expected-count penalty still pushes the gate toward closure.
5. As the gate probability falls, it receives fewer future physical-gradient opportunities.

The previous architecture also gave group 0 a special role:

\[
z_{b0}=1
\]

with no count cost, while all other groups were penalized. This creates a natural sink: the rich fine QM/QE readers can route everything through the free group even when doing so hurts early accuracy.

The observed result is exactly that failure mode: rank-1 routing, no fine-work reduction, and lower fidelity.

The remedy should therefore address **training dynamics and symmetry**, not add more sparsity penalties.

---

# 2. What should remain unchanged

Keep the Run-1409 scientific model:

- `forward_architecture="budgeted_group_control_honf"`;
- `Kmax=12`;
- control width `D=16`;
- Dense-1804 MM/ME/EM preparation;
- Run-1406 group-conditioned fine QM/QE formulation;
- collapsed query-source relation
  \[
  \rho^S_{qs}=\sum_k \alpha_{qk}A^S_{sk};
  \]
- collective moment
  \[
  n^S_{qs}=\sum_k\alpha_{qk}A^S_{sk}h_k;
  \]
- one expensive fine interaction per physical query-source pair;
- `C_g+C_M+C_E`;
- P0/P1/P2 physical coupling;
- predicted ports;
- optimizer, base learning rate, data, loss terms, and seed;
- compact/full-width executor parity infrastructure.

Do not introduce:

- pair-cost losses;
- balance losses;
- entropy losses;
- target group-count losses;
- geometry barriers;
- top-k routing;
- sampled QE;
- custom kernels;
- dynamic sequential group generation.

The only added training pressure remains one case-level expected group-count cost.

---

# 3. Do not initialize A and alpha as exactly fixed uniform tensors

The intuition "start uniform, then compact" is correct, but literal fixed uniform assignments are undesirable.

If every group receives exactly the same assignment and the model has no symmetry-breaking signal, all groups can receive nearly identical gradients and remain redundant.

Instead use:

\[
\boxed{\text{near-uniform, differentiable, symmetry-broken routing}}
\]

during the initial organization stage.

Keep the learned group prototypes. Make their routing logits weak at the beginning and continuously restore the full logits.

For source or query logits \(\ell\), define

\[
\boxed{
\ell^{(\mathrm{route})}(t)
=
r(t)\,\ell
}
\]

before entmax, where

\[
r(1)=0.1,\qquad r(t)=1
\]

by the end of the organization warm-up.

With all gates open and \(r\ll1\), entmax receives near-equal logits and produces almost-uniform memberships, but gradients still flow because \(r>0\).

Use a simple linear continuation:

\[
r(t)
=
0.1+0.9
\min\left(\frac{t-1}{24},1\right).
\]

Thus epochs 1–25 move smoothly from near-uniform routing to the ordinary learned Run-1409 routing logits.

No separate softmax implementation is required.

---

# 4. Remove the privileged always-free group

All Kmax groups should be exchangeable.

Replace the permanent

\[
z_0=1
\]

with symmetric gates for all \(K_{\max}=12\) groups.

The system still needs a defined result if every hard gate closes. Use a numerical fallback only:

\[
k^\star=\arg\max_k a_k,
\]

and if a hard deterministic/stochastic support is empty, activate only \(k^\star\).

This fallback:

- is not an additional learned branch;
- is not a special group identity;
- is used only to make normalization well-defined;
- should be reported whenever it occurs.

The group-count objective should charge only **excess capacity above one group**, symmetrically:

\[
\boxed{
L_K
=
\lambda(t)\,
\mathbb E_b
\left[
\max\left(0,\sum_k p_{bk}-1\right)
\right].
}
\]

Subtracting the free baseline in this symmetric way avoids a permanently privileged prototype.

---

# 5. Dense-to-sparse gate continuation

The critical change is that hard-concrete gates must not be allowed to remove a group's physical gradient before useful organization has formed.

Let the ordinary hard-concrete sample/deterministic value be

\[
\widetilde z_{bk}\in[0,1].
\]

Define a continuation coefficient \(c(t)\).

### Stage A — organization warm-up: epochs 1–25

\[
c(t)=0.
\]

Use

\[
\boxed{z^{\mathrm{eff}}_{bk}=1.}
\]

for all groups.

There is no group-count loss:

\[
\lambda(t)=0.
\]

The gate network may compute logits for diagnostics, but those logits do not affect the predictor.

This stage learns a 12-group organization before compression begins.

### Stage B — gradual compression: epochs 26–150

Linearly increase

\[
c(t)
=
\frac{t-25}{125}
\]

clipped to \([0,1]\).

Use

\[
\boxed{
z^{\mathrm{eff}}_{bk}
=
(1-c(t))
+
c(t)\widetilde z_{bk}.
}
\]

For every \(c<1\),

\[
z^{\mathrm{eff}}_{bk}>0,
\]

so every group retains a physical gradient path even if the underlying hard-concrete sample would otherwise be zero.

Ramp the existing final count coefficient with the same continuation:

\[
\boxed{
\lambda(t)
=
c(t)\lambda_\star.
}
\]

Use the previous calibrated final coefficient

\[
\lambda_\star
=
0.005783974924700852.
\]

Keeping the final coefficient unchanged isolates the effect of the optimization path rather than introducing a second sparsity-strength experiment.

### Stage C — fully budgeted model: epoch 150 onward

Set

\[
c=1,
\qquad
z^{\mathrm{eff}}=\widetilde z,
\qquad
\lambda=\lambda_\star.
\]

Now exact group closure is permitted.

At this point the learned structure, rather than an untrained gate prior, must decide which groups survive.

---

# 6. Effective support during the continuation

During Stage A and while \(c<1\), all groups are physically available:

\[
g_{bk}=1.
\]

Do not pack columns simply because the underlying hard-concrete sample is zero: the continuation term gives the group a positive effective gate and therefore a live physical contribution.

Only when \(c=1\) can

\[
z^{\mathrm{eff}}_{bk}=0
\]

produce an exact closed column.

This means:

- epochs <150 are intentionally not execution-sparse;
- their purpose is to learn a stable factorization;
- runtime savings should only be expected after full hardening.

Do not misreport the raw hard-concrete support as executed active K during the continuation.

Track both:

\[
K_{\mathrm{raw}}=\sum_k\mathbf 1[\widetilde z_k>0]
\]

and

\[
K_{\mathrm{executed}}=\sum_k\mathbf 1[z^{\mathrm{eff}}_k>0].
\]

---

# 7. Kappa normalization remains capacity-invariant

Retain the gate-reference normalization, but compute it from the **effective** gate:

\[
\eta_b
=
\operatorname{entmax}_{1.5}
(\log z_b^{\mathrm{eff}})
\]

over its positive support, and

\[
\boxed{
\kappa_b
=
\frac{1}{\sum_k\eta_{bk}^2}.
}
\]

During the all-open warm-up,

\[
z^{\mathrm{eff}}=1
\Rightarrow
\eta_k=\frac1{K_{\max}}
\Rightarrow
\kappa=K_{\max}=12.
\]

As groups are compressed, \(\kappa\) changes continuously until exact closure is enabled.

Keep the Run-1409 rule that capacity, packed width, live count, numerical rank, and \(\kappa\) are separate quantities.

---

# 8. Why this is a better test than lowering lambda

Simply reducing \(\lambda_\star\) would make collapse slower, but it would not determine whether the original failure was due to premature closure or to the final budgeted objective itself.

The proposed rerun preserves the same final penalty and asks a cleaner question:

> If the model first learns a useful multi-group representation, will physical gradients preserve the groups that are actually valuable when compression is introduced gradually?

Possible outcomes are interpretable.

### Outcome A — multiple groups survive and accuracy approaches 1406

The first Run 1409 failed because it compressed too early.

### Outcome B — the model learns well during warm-up but collapses to K=1 after full hardening and accuracy degrades

Then the final group-count objective itself prefers rank-1 routing for this task. Further scheduling is unlikely to fix the scientific formulation.

### Outcome C — multiple groups survive but fine query-source support stays dense

Then adaptive rank works, but it does not create adaptive fine computation.

### Outcome D — group count and fine support both shrink with good accuracy

This is the desired result and justifies executor work.

---

# 9. Coding changes

Keep changes local to the existing Run-1409 implementation.

Primary files:

```text
src/honf_forward_core/interface_fields/case_group_budget.py
src/honf_forward_core/interface_fields/budgeted_group_control.py
src/honf_forward_core/config.py
src/config_core/forward/budgeted_group_control_honf_context.json
Case_ThermalChannel/.../training loss integration
```

Do not fork another backend.

## 9.1 CaseGroupGate

Change the gate from `K-1` optional logits plus fixed group 0 to K symmetric logits.

Add a helper that returns:

- raw hard-concrete gate \(\widetilde z\);
- positive probabilities \(p\);
- effective continuation gate \(z^{eff}\);
- raw support;
- executed support;
- fallback-used flag.

The group-count loss must expose a live tensor:

\[
\max(0,\sum_k p_k-1).
\]

## 9.2 Training progress

The architecture needs the current epoch only to compute two deterministic scalars:

- routing logit strength \(r(t)\);
- sparsification continuation \(c(t)\).

Use the existing training-progress call path if available.

Do not add a scheduler service or separate training loop.

Store no new optimizer state for the continuation itself.

## 9.3 Router

Before module/environment/query entmax:

\[
\ell\leftarrow r(t)\ell.
\]

Then add the effective gate prior:

\[
\ell\leftarrow\ell+\log z^{eff}.
\]

During \(c<1\), every group remains in the mask.

At \(c=1\), use exact support from the hard-concrete gate.

## 9.4 Compact executor

Before full hardening, compact mode should have width 12 because all effective gates are positive.

After full hardening, pack only exact positive columns as in the current implementation.

Do not attempt a selected fine QE reader unless measured unique source support becomes materially sparse.

The first goal is to establish adaptive capacity without sacrificing fidelity.

---

# 10. Initialization and sanity checks

Use the existing trainable group prototypes; do not zero them.

Near-uniform initial routing comes from \(r(1)=0.1\), not from making all group parameters identical.

Initialize all gate logits to the existing high-open policy. A target initial positive probability of 0.95 is acceptable because Stage A forces the predictor to all-open anyway.

Before the managed run, use two real training batches and verify by **actual forward execution**:

- module/environment/query normalized routing entropy is high at epoch 1;
- every effective gate is positive;
- all 12 groups receive finite physical gradients;
- group/prototype gradients are not numerically identical across all columns;
- one real predicted-port optimizer update is finite.

Do not tune initialization by a broad sweep.

If the routing is not near-uniform despite \(r=0.1\), report the measured logits/entropy and make one narrowly justified adjustment to the initial routing scale before launch.

---

# 11. Fresh managed Run 1409 rerun

Launch a new timestamped run using logical ID:

```text
1409
```

Suggested name:

```text
case_budgeted_group_control_dense_to_sparse_rerun
```

Do not resume the collapsed first Run 1409.

Use the same:

- seed 0;
- learning rate 3e-4;
- AdamW setup;
- physical loss;
- predicted ports;
- dataset split;
- Kmax=12;
- D=16.

Train to epoch 50 first.

---

# 12. Epoch-50 review

Epoch 50 occurs during gradual compression, so **do not require hard K sparsity yet**.

Compare to exact-epoch-50 Run 1406 and the first Run 1409.

Review:

### Learning

- validation field MSE;
- trailing-10 validation median;
- field/temperature losses;
- finite gradients/updates.

A useful target is to recover most of the parent learning speed:

\[
\mathrm{MSE}_{1409v2,50}
\lesssim1.2\,
\mathrm{MSE}_{1406,50}.
\]

This is a research review band, not an embedded software gate.

### Organization

Report:

- raw expected K;
- raw hard-concrete support;
- executed K (expected to remain 12 while \(c<1\));
- per-case gate-logit variance;
- source/query effective groups;
- routing matrix numerical ranks.

The failure mode to avoid is already visible if all optional probabilities have collapsed near zero by epoch 50 even though the continuation still keeps them executable.

### Cost

Measure training step and memory, but do not expect inference acceleration before hardening.

If training is healthy and the gate probabilities remain differentiated rather than universally collapsed, continue the same run.

---

# 13. Epoch-150 hardening review

At or just after epoch 150, exact closure is enabled.

Run a compact audit on all 90 cases:

- deterministic \(K_{\mathrm{live}}\);
- expected K;
- distribution of active prototype identities;
- fraction of fallback cases;
- query/source effective groups;
- routing ranks;
- unique QM/QE support;
- actual executed rows;
- compact controller width;
- prediction error.

The model must not be described as adaptive K unless deterministic live count varies across cases or, at minimum, different cases select meaningfully different live prototype subsets.

If it again collapses to one group for all cases immediately after hardening with a substantial accuracy loss, stop and report. Do not introduce another schedule or penalty in the same run.

---

# 14. Continuation to epoch 500

If the epoch-150 state remains physically competitive and nontrivial, continue the same run and optimizer to epoch 500.

At epoch 500 perform the matched 90-case comparison against:

- Run 1406;
- first Run 1409;
- Dense 1804;
- Run 1404 for latency context.

Report separately:

\[
K_{\max},
\quad
K_{\mathrm{expected}},
\quad
K_{\mathrm{live}},
\quad
K_{\mathrm{pack}},
\quad
\operatorname{rank}(R^M),
\quad
\operatorname{rank}(R^E).
\]

And also:

\[
P_M^{support},\quad
P_E^{support},\quad
P_M^{executed},\quad
P_E^{executed}.
\]

Measure actual latency and memory.

A lower K with dense fine support is adaptive representation, not adaptive physical computation.

Do not automatically extend to 5000 epochs. Wait for user review.

---

# 15. Tests

Add only focused tests needed by the revised science:

1. symmetric gate permutation behavior;
2. all-open warm-up reduction;
3. continuation gate formula;
4. no exact closure for \(c<1\);
5. symmetric all-closed fallback;
6. gate-loss counted once;
7. effective-gate kappa normalization;
8. full/compact parity after hardening;
9. mixed-case batch invariance;
10. checkpoint resume restores training progress and RNG;
11. historical Run-1409/1406 strict loading remains unchanged where applicable;
12. one real predicted-port update.

Do not create a new test framework or broad defensive infrastructure.

---

# 16. Required report

Create:

```text
docs/reports/HONF_Run1409_DenseToSparse_Rerun_Evaluation.md
```

The report should answer:

1. Did the warm-up prevent premature case-rank collapse?
2. Did group roles differentiate before compression?
3. What happened when exact hardening began?
4. Does deterministic K vary by case?
5. Does lower K reduce unique physical pair support?
6. Does lower support reduce actual executed rows?
7. What accuracy is retained relative to Run 1406?
8. Does compact K improve controller cost in practice?
9. Is the remaining limitation adaptive-rank learning or fine-reader sparsity?

Preserve existing security/trusted loading.

Use ordinary Git, focused tests, and actual GPU execution.

Do not add hashes, freezes, monitoring services, baseline-copy infrastructure, or other defensive process machinery.
