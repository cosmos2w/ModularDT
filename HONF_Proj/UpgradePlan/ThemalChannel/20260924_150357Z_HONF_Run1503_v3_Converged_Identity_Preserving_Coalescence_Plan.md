# HONF Run 1503-v3 — Converged, Identity-Preserving Reversible Coalescence

## Goal

Continue the HONF development from the completed `1503-v2 / reversible_edge_coalescence` experiment (managed numeric Run 1504) without continuing that stopped checkpoint.

The central scientific goal for this round is:

> **Remove redundant query-access / interaction functions while preserving the fine physical information they organize.**

This is a formation experiment first. It is **not** a new environmental pooling experiment, not a sparsity-normalizer sweep, and not a new executor-kernel project.

The scientific base remains Run 1502:

- Dense-style fine MM/ME/EM preparation;
- final environmental sparsemax refinement;
- module entmax-1.5 assignment;
- K=12 provisional registered capacity;
- D=16 group control;
- prototype-anchored RMS query keys;
- query sparsemax;
- source-resolved many-body controls;
- one fine physical interaction per query-source pair;
- three-term field form `C = C_g + C_M + C_E`.

The successful pieces of Run 1503-v2 are retained:

- multiplicity-correct quotient routing;
- merged source incidence `Abar`;
- source-resolved control moment `B`;
- provisional `kappa_0`;
- original contextualized physical source states and K/V rows;
- the corrected executor dispatch;
- exact quotient tests;
- optional fused CSR executor as an evaluation-only reference.

The failed part is changed:

> **The convex-clustering solution must no longer replace every access descriptor before an actual merge occurs.**

A case/phase with no accepted merge must reduce exactly to the Run-1502 operator.

---

# 1. Why Run 1504 should not be resumed

Run 1504 established several useful facts.

1. The quotient mathematics is sound.
   - weighted sparsemax matches the virtual repeated-group representation;
   - `Abar` and heterogeneous source-control moment `B` are conserved;
   - true-merge forward and first-gradient parity tests passed.

2. The numerical solver was not reliable at the fixed 64-step production setting.
   - real trained phases had primal residuals as large as `1e-2`;
   - 128–256 iterations changed several real partitions;
   - partitions were stable after convergence in the documented examples.

3. More importantly, **the forward operator was changed before any discrete coalescence happened**.
   - the 64-step endpoint produced zero non-singleton classes across the 270 surveyed case-phase plans;
   - nevertheless its field reconstruction was already substantially worse than same-stage Run 1502;
   - frozen and e50 diagnostics showed nonzero access-descriptor displacement even when all classes remained singleton.

Therefore a longer solver is necessary but not sufficient.

Changing only 64 → 256/512 iterations in the existing Run-1504 checkpoint would change its operator and would not repair the more fundamental singleton-path problem. Do not resume Run 1504.

Launch a new separately identified experiment after the changes below.

---

# 2. Scientific hypothesis

Run 1502 contains a learned set of twelve **provisional interaction functions** per case and phase.

Some provisional functions may become redundant for a particular physical case. The model should be allowed to quotient those functions into a smaller case-local hypergraph, but only when the clustering solution actually identifies a fused class.

The new hypothesis is:

\[
\boxed{
\text{provisional K=12 organization}
\;\longrightarrow\;
\text{converged reversible partition}
\;\longrightarrow\;
\text{R case-local interaction functions}
}
\]

with

\[
1 \le R \le 12,
\]

while every original fine module/environment source and its physical state remains available.

The desired behavior is not a prescribed target such as `R=6`.

A valid result may have:

- R=12 for difficult cases;
- R=10–11 for moderately redundant cases;
- smaller R only when the learned functions actually support it.

The number of groups should emerge from the learned case/phase organization and the fixed fusion strength, not from a K-count loss or deletion gate.

---

# 3. Keep the convex-clustering objective

Retain the existing access descriptor construction and pair weights unless a code-level correctness issue is found.

For active provisional groups \(k=1,\ldots,K_a\), let

\[
z_k \in \mathbb{R}^{d_z}
\]

be the existing query-access descriptor.

Retain the same convex clustering problem:

\[
\boxed{
U^\star
=
\arg\min_U
\frac12\sum_k \|u_k-z_k\|_2^2
+
\eta(e)\sum_{k<l} w_{kl}\|u_k-u_l\|_2
}
\]

with the same experiment strength:

\[
\eta(e)=0.5\min(e/150,1).
\]

Do not change `eta_final=0.5` in this experiment.

Do not introduce:

- a K target;
- count penalty;
- hard-concrete group gate;
- source/query top-k;
- a second normalizer;
- a second seed;
- an eta sweep.

The purpose is to isolate whether the **same mathematical fusion idea succeeds when solved correctly and inserted into the forward operator correctly**.

---

# 4. Solver revision: solve the stated problem to convergence

## 4.1 Remove the fixed-64-step scientific semantics

`64 steps` must no longer define the model.

ADMM iteration count is a numerical implementation detail. The model is the solution of the convex problem to a documented numerical tolerance.

Use standard primal/dual residual stopping.

For the current split formulation, compute the actual ADMM primal and dual residuals used by the implementation and terminate a case only when both satisfy absolute-plus-relative criteria:

\[
\|r^t\|_2 \le \epsilon_{\rm pri},
\qquad
\|s^t\|_2 \le \epsilon_{\rm dual}.
\]

Use the standard scale-aware form

\[
\epsilon_{\rm pri}
=
\sqrt{n_r}\epsilon_{\rm abs}
+
\epsilon_{\rm rel}
\max(\|D U\|_2,\|V\|_2),
\]

\[
\epsilon_{\rm dual}
=
\sqrt{n_s}\epsilon_{\rm abs}
+
\epsilon_{\rm rel}
\|\rho D^\top Y\|_2,
\]

adapted exactly to the tensors/dual convention in the current solver.

Initial implementation values:

```text
eps_abs = 1e-6
eps_rel = 1e-5
max_iterations = 512
```

These are numerical tolerances, not learned architecture hyperparameters.

Retain the current ADMM penalty parameter initially. Do not add adaptive-rho logic unless actual execution shows it is necessary.

## 4.2 Solve the tiny clustering problem in FP64

The clustering problem is extremely small compared with the neural physical operator: at most 12 functions and the current low-dimensional access descriptor.

Perform the ADMM state and residual arithmetic in FP64 on the same GPU.

Do not change the neural model dtype.

Do not move the solver to CPU, because CPU/GPU synchronization would introduce unnecessary training overhead.

## 4.3 Planner does not need autograd through ADMM

The coalescence partition is a discrete, piecewise-constant structural plan.

Run the numerical clustering solve under `torch.no_grad()` / detached descriptors.

Backpropagation should operate through the **physical compact representation for the chosen partition**, not through hundreds of numerical solver iterations.

This removes:

- ADMM autograd memory;
- sensitivity of physical gradients to an incompletely converged unrolled optimizer;
- the need for the solver itself to be a differentiable layer.

The model remains differentiable inside a fixed partition, exactly as sparsemax has piecewise support.

## 4.4 Nonconvergence must fall back to the parent structure

If a particular case/phase has not converged by 512 iterations:

- do not use its unconverged pseudo-clusters;
- use the singleton parent plan for that case/phase;
- record `solver_converged=False`, final residuals, and iteration count.

Do not abort ordinary training solely because one tiny clustering solve did not converge.

This is an algorithmic definition of a safe unresolved case, not a software release gate.

## 4.5 Class extraction

Keep the current class tolerance initially:

```text
class_tolerance = 1e-5
```

but apply it **only after the ADMM convergence condition is satisfied**.

Class decisions must therefore no longer depend on whether the arbitrary loop happened to stop at step 64.

For a fixed small diagnostic panel, verify that:

- the partition at convergence agrees with a longer replay;
- halving `class_tolerance` does not broadly rewrite the partition;
- cases that fail this numerical stability check are reported, not relabeled.

Do not tune the tolerance to manufacture a desired R histogram.

---

# 5. Central operator change: singleton identity

This is the most important change in the new experiment.

The converged convex solution \(U^\star\) determines **which provisional groups belong to one class**.

It must not automatically replace every provisional access descriptor.

Let the partition be

\[
\mathcal C
=
\{C_1,\ldots,C_R\}.
\]

## 5.1 Singleton class

For

\[
C_r=\{k\},
\]

the compact access function must use the original Run-1502 quantities for group \(k\).

No proximal displacement is allowed.

Therefore, if every class is singleton,

\[
R=K_a,
\]

the complete query router and physical operator must be the Run-1502 parent operator.

This property must hold at nonzero eta.

The intended invariant is:

\[
\boxed{
\text{no accepted merge}
\Longrightarrow
\text{parent routing and physical prediction}
}
\]

not merely “approximately the parent”.

## 5.2 Merged class

Only for

\[
|C_r|>1
\]

construct one compact access function.

Do **not** use the raw ADMM centre as a synthetic physical centroid.

The solver decides the partition; the forward operator should construct the merged access function from the original learned physical organization.

---

# 6. Physically grounded merged access function

For class \(C_r\), retain the current multiplicity

\[
m_r=|C_r|.
\]

## 6.1 Merge source incidence exactly

For source type \(S\in\{M,E\}\),

\[
\boxed{
\bar A^S_{sr}
=
\sum_{k\in C_r} A^{S,0}_{sk}
}
\]

as already implemented.

Because provisional source rows are normalized across K,

\[
\sum_r \bar A^S_{sr}=1.
\]

Do not pool the fine source state itself.

## 6.2 Preserve the source-resolved control moment

Retain

\[
\boxed{
B^S_{sr}
=
\sum_{k\in C_r}
A^{S,0}_{sk}h^0_k.
}
\]

The query-source many-body control remains

\[
n^S_{qs}
=
\sum_r b_{qr}B^S_{sr}.
\]

This prevents destructive averaging of heterogeneous constituent controls.

Retain all existing quotient/moment tests from v2.

## 6.3 Recompute merged physical centres from merged incidence

Do not take a moved ADMM descriptor centre as the physical group location.

For modules:

\[
\mu^M_r
=
\sum_i \nu^M_i\bar A^M_{ir},
\]

\[
c^M_r
=
\frac{
\sum_i \nu^M_i\bar A^M_{ir}x_i
}{
\mu^M_r+\epsilon
}.
\]

For environment sources:

\[
\mu^E_r
=
\sum_j \nu^E_j\bar A^E_{jr},
\]

\[
c^E_r
=
\frac{
\sum_j \nu^E_j\bar A^E_{jr}x_j
}{
\mu^E_r+\epsilon
}.
\]

Use the same source-type fallback logic as Run 1502 when one type has zero mass.

This ensures that the compact hyperedge geometry still describes the fine sources that it organizes.

## 6.4 Merge query keys by barycentric representation

Let \(t_k\) be the existing normalized Run-1502 group query key.

For a merged class use

\[
\tilde t_r
=
\operatorname{RMSNorm}
\left(
\frac{1}{m_r}\sum_{k\in C_r} t_k
\right).
\]

For singleton classes use the original \(t_k\) directly, without re-evaluating an alternative arithmetic path.

The compact query logit is then built using the same Run-1502 formula:

\[
L_{qr}
=
\frac{
\tilde q_q^\top\tilde t_r
}{
\sqrt D
}
+
g(q,c^M_r,c^E_r).
\]

No new query network is introduced.

---

# 7. Multiplicity-correct compact routing

Keep the weighted sparsemax quotient developed in v2.

For class multiplicity \(m_r\),

\[
a_{qr}
=
m_r [L_{qr}-\tau_q]_+,
\]

with

\[
\sum_r a_{qr}=1.
\]

`a` is the class probability mass.

Define per-constituent density

\[
\boxed{
b_{qr}
=
\frac{a_{qr}}{m_r}.
}
\]

The physical contractions use `b`:

\[
\boxed{
\rho^S_{qs}
=
\sum_r b_{qr}\bar A^S_{sr},
}
\]

\[
\boxed{
n^S_{qs}
=
\sum_r b_{qr}B^S_{sr}.
}
\]

This remains exactly equivalent to a virtual expanded model in which every constituent of a merged class shares the same compact fused logit.

Keep the existing unequal-multiplicity forward/gradient tests.

---

# 8. Preserve parent amplitude calibration

Keep:

- provisional `kappa_0`;
- literal registered `K=12` finalization factors;
- the current value-control reconstruction from the conserved source moment;
- the existing source physical measures.

Do not recompute kappa from the compact R groups in this experiment.

A successful merge is a quotient of interaction functions, not a new physical-mass normalization rule.

---

# 9. Fine physical information must remain untouched

The new 1503-v3 model must still retain:

- every contextualized fine module state;
- every contextualized fine environmental state;
- every environmental K/V row;
- every source coordinate;
- every source quadrature measure;
- fine module relative geometry;
- fine environmental query/source geometry;
- source-local value modulation;
- the original one-softmax environmental response over its physical source support.

Do not introduce:

- pooled field values;
- group-value decoding;
- regional replacement of fine sources;
- adaptive quadrature;
- a coarse environmental bypass;
- a new local correction branch.

The hypergraph compresses the **interaction organization**, not the physical state.

---

# 10. Do not add a new coalescence loss in this run

The first revised experiment should not introduce an auxiliary K/count/fusion loss.

Reason:

The previous run does not tell us whether converged reversible coalescence itself is bad, because its numerical solve was incomplete and it changed singleton access functions continuously.

The clean next experiment should answer that question first.

If the converged, identity-preserving model still yields almost no mergers at epoch 150/500, then the next research question becomes whether the learned provisional functions need an explicit redundancy-inducing objective.

Do not answer that second question in the same run.

---

# 11. Code implementation route

Codex should first inspect the exact files introduced by commit `ae3e0fa` and reuse the existing 1503-v2 implementation rather than starting a parallel stack.

The expected changes are conceptually:

## A. Convex coalescence solver

Modify the existing ADMM solver to:

- detached FP64 state;
- residual-based convergence;
- max 512 steps;
- return per-case:
  - converged flag;
  - iteration count;
  - primal residual;
  - dual residual;
  - converged fused centres;
  - class IDs.

Keep the current objective and pair weights.

## B. Planner / prepared state

The plan must retain:

- provisional group IDs;
- class membership;
- multiplicities;
- whether each class is singleton;
- solver diagnostics.

For nonconverged cases return singleton classes.

## C. Compact control construction

Reuse current:

- `Abar`;
- `B`;
- multiplicity packing;
- quotient routing.

Change:

- merged geometry to incidence-derived physical centres;
- merged query keys to original-key barycentres;
- singleton query keys/centres to the original parent tensors exactly.

## D. Parent-exact singleton fast path

When a prepared case/phase contains only singleton classes, call the normal Run-1502 router/control path directly where practical.

At minimum, ensure output and first gradients are numerically identical to the parent path.

Do not run the ADMM fused descriptors through the ordinary query forward in this branch.

## E. Executor

Do not redesign the executor in this experiment.

- rectangular remains production/default;
- corrected `support_blocks` remains a reference;
- fused CSR remains evaluation-only.

Formation and execution evidence must remain separate.

---

# 12. Required tests before training

## 12.1 Solver convergence

Use the real trained descriptors already identified in the previous report:

- 0273;
- 0686;
- 0281;
- 0277;
- 0676 where available.

Verify that the new residual-stopped solver reproduces the stable 128/256/512 partitions reported previously.

For case 0281 P0, verify that the solver actually continues past the old 64-step state until its residual criterion is met.

## 12.2 Singleton parent identity at nonzero eta

This is mandatory.

Construct real prepared phases with:

```text
eta > 0
R = K_occupied
all classes singleton
```

Verify against Run 1502:

- query logits;
- sparsemax route;
- rho;
- n;
- QM context;
- QE context;
- full field;
- first gradients for:
  - query coordinates;
  - module coordinates;
  - group codes;
  - source assignment/control parameters.

The singleton path must not reproduce the old v2 descriptor drift.

## 12.3 Controlled merge quotient

Retain and extend the current controlled two-member merge test.

Check:

- Abar;
- B;
- merged physical centres;
- multiplicity-correct sparsemax;
- virtual-expanded rho/n equivalence;
- QM/QE result;
- first gradients.

## 12.4 Real merge replay

On an existing checkpoint where converged ADMM proposes a real two-member class:

- execute the actual compact operator;
- compare against the corresponding virtual expanded fused operator;
- record prediction change relative to the unmerged parent operator separately.

Do not confuse quotient exactness with parent equivalence.

## 12.5 Numerical boundary

Repeat the physical-coordinate sweep around at least one observed merge/split boundary if one can be found.

Report:

- class identity;
- output continuity;
- local AD vs centered finite differences away from the transition;
- actual jump magnitude at the transition if present.

Do not claim global design smoothness from fixed-partition autograd.

## 12.6 Real optimizer step

Execute ordinary predicted-port training steps on:

- mixed low-M batch;
- M=10 batch.

Require finite:

- loss;
- gradients;
- parameters;
- optimizer state.

Record actual solver iteration distribution and solver wall time.

No disposable state may be reused for the managed run.

---

# 13. Bounded frozen diagnostics before the new formal run

Use existing static checkpoints only.

## Mature Run 1502

On the mature Run-1502 best-field checkpoint:

- build converged plans at final eta=0.5;
- report R distributions for P0/P1/P2;
- execute the identity-preserving compact operator;
- measure frozen prediction perturbation;
- measure solver iterations/residuals.

This answers whether mature 1502 already contains removable redundant access functions.

## Stopped Run 1504 e149/e150

Replay the new converged planner on the old trained descriptors only for diagnosis.

Do not relabel those checkpoints as repaired models.

Compare:

- old fixed-64 partition;
- converged partition;
- new identity-preserving frozen output.

This localizes how much of the previous degradation came from pre-merge descriptor displacement versus training trajectory.

These are diagnostics, not substitutes for a fresh training run.

---

# 14. Formal training experiment

Launch one new managed run from scratch using the allocator's next numeric Run ID.

Suggested conceptual name:

```text
1503_v3_converged_identity_preserving_coalescence
```

Use the Run-1502 science/training base:

- same seed;
- same 600/90 split;
- same optimizer;
- same learning rate and weight decay;
- same losses;
- same predicted-port policy;
- same K=12;
- same D=16;
- same source/query normalizers;
- same fine physical preparation;
- same receiver chunk for training.

Only differences from Run 1502 should be:

- the new architecture identifier;
- converged reversible planner;
- same eta=0.5 / ramp=150 fusion setting;
- compact quotient path when an actual class merger exists.

Do not:

- warm start from Run 1504;
- warm start from mature Run 1502;
- launch another eta;
- launch another seed;
- add a fusion loss;
- change K;
- change sparsemax/entmax;
- change the executor used for training.

---

# 15. Epoch-50 review

At epoch 50 eta is still one third of its final value.

Inspect:

## Learning

- train/validation trajectory;
- finite parameters/optimizer;
- actual update norms.

## Solver

- fraction of converged plans;
- solver iteration distribution;
- max/median normalized residual;
- solver wall-time contribution.

## Formation

For all 90 development cases with deterministic Q1024 formation evidence:

- R per case/phase;
- number of true non-singleton classes;
- multiplicity distribution;
- Kq in provisional and compact coordinates;
- occupied proposal K;
- compact R;
- module/environment source degree;
- unique query-source support.

## Parent-tracking diagnostic

For cases/phases with no merges, the candidate should behave as the parent operator.

If the model is materially behind the Run-1502 e50 trajectory while essentially all plans remain singleton, investigate implementation/numerical differences before proceeding.

This is a scientific diagnosis, not an automated scalar gate.

If the model is healthy, continue to epoch 150.

---

# 16. Epoch-150 review

At epoch 150 the fusion strength reaches eta=0.5.

Perform:

- full 90-case Q1024 formation;
- matched 90-case Q8192 full-grid evaluation against Run 1502 e150;
- exact endpoint and saved-best-field candidate checkpoints separately.

Answer explicitly:

1. Did convergence-stable true coalescence occur?
2. Is R genuinely case dependent?
3. Are mergers phase dependent?
4. Are singleton cases still parent-like?
5. Does physical accuracy remain competitive?
6. Did near/far and thermal/interface behavior remain balanced?
7. Did any merge create a large design-output discontinuity?

A successful formation result does not require a specific mean R.

But `all cases R=12` means the formation objective has not yet succeeded.

If the model is numerically stable, has real convergence-stable mergers, and its physical trajectory remains credible, continue unchanged to epoch 500.

Otherwise stop and report the reason.

---

# 17. Epoch-500 decision

Stop at 500 for explicit user review.

Evaluate:

- exact e500;
- saved best field;
- saved best total if different;
- full 90-case Q8192 reconstruction;
- Q1024 formation at the same checkpoint used for each accuracy claim;
- per-case paired differences;
- active-module strata;
- near/far;
- fluid channels;
- internal/surface temperature;
- heat flux;
- port temperature/effective h.

Also measure:

- complete application;
- preparation;
- prepared P2 decode;
- solver time;
- rectangular work;
- optional fused CSR work/time/memory.

Do not combine accuracy from one checkpoint with R/support from another.

Recommend a 5000-epoch maturation only if:

- mergers are convergence-stable and genuinely case dependent;
- the accuracy trajectory is competitive enough to justify maturation;
- no broad physical-output failure appears;
- design sweeps do not reveal unacceptable merge-boundary jumps.

No automatic continuation beyond 500.

---

# 18. Execution work in this round

The executor result from v2 is informative:

- selected QE geometry rows fell by about 53%;
- allocation fell materially;
- complete application remained slightly slower.

Therefore do not make a new kernel the central task here.

Retain the corrected fused CSR prototype and benchmark it only as supporting evidence.

Do not:

- add another Triton kernel;
- change numerical precision of the physical attention;
- add block/signature execution;
- change training to selected execution;
- claim speed from row reduction.

Once formation succeeds, the next executor round can profile the now-smaller structural/control work together with physical support and determine whether to:

- fuse geometry MLP work differently;
- change selected-source packing;
- compile the complete selected reader;
- or retain dense execution for small shapes.

---

# 19. Code cleanup policy

Do not delete historical architectures during this experiment.

However, new 1503-v3 production code should conceptually depend only on:

1. Run-1502 sparse-incidence source organization;
2. converged reversible case/phase planner;
3. compact quotient router;
4. conserved source/control moments;
5. existing fine QM/QE physical readers.

The following should not become part of the new core:

- adaptive-opening coarse field path;
- direct group-value field decoding;
- case-level alive/dead gates;
- K-count penalties;
- top-k routing;
- early pooled physical source states;
- query-specific quadrature banks.

Historical code may remain for reproducibility.

---

# 20. Software-process constraints

This is research development.

Use ordinary:

- Git history;
- configs;
- focused tests;
- real GPU execution;
- existing run/checkpoint machinery.

Do not add:

- cryptographic hashes;
- contract freezes;
- baseline snapshot systems;
- approval databases;
- monitoring daemons;
- extra blocking infrastructure.

Preserve all existing checkpoint/security behavior.

Numerical solver stopping is part of the mathematical algorithm, not a production-release gate.

Preflight/unit tests do not replace:

- real optimizer execution;
- formal training;
- full-grid physical evaluation;
- synchronized timing.

---

# 21. Definition of done

This round is complete when:

1. the fixed-iteration ADMM semantics are replaced by residual-converged FP64 planning;
2. solver iterations are removed from the autograd graph;
3. nonconverged plans safely remain singleton;
4. singleton classes at nonzero eta reproduce the Run-1502 parent operator;
5. merged classes preserve Abar/B quotient mathematics and original fine physical sources;
6. merged geometry comes from merged physical incidence rather than arbitrary ADMM coordinate displacement;
7. focused CPU/GPU tests pass;
8. real predicted-port optimizer steps pass;
9. frozen mature-parent diagnostics are executed;
10. one fresh formal candidate is launched;
11. it is reviewed at epochs 50 and 150;
12. if justified, it runs to at most epoch 500;
13. a final report distinguishes:
    - numerical convergence,
    - actual edge coalescence,
    - physical fidelity,
    - design-gradient behavior,
    - executed computation.

The central success criterion is:

\[
\boxed{
\text{fewer case-local interaction functions}
\quad\text{without discarding the fine physical information they organize.}
}
\]

The key falsifiable property is equally important:

\[
\boxed{
\text{if nothing is redundant and nothing merges, the model must remain the Run-1502 operator.}
}
