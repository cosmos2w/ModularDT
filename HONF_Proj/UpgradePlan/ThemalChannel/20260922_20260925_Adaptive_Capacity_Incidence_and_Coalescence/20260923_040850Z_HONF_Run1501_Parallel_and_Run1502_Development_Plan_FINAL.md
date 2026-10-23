# HONF Run 1501 Parallel Development + Run 1502 Workflow Override

## Document relationship and precedence

Keep exactly **two** planning documents for this task:

1. `HONF_Run1501_Maturation_and_Run1502_Development_Plan.md`
   - scientific and mathematical reference;
   - Run-1501 formulation and notation;
   - support-union executor equivalence requirements;
   - scientific rationale and intervention definitions;
   - Run-1502 mathematical hypothesis;
   - evaluation and claim/evidence standards.

2. `HONF_Run1501_Parallel_and_Run1502_Development_Plan_FINAL.md`
   - authoritative current workflow;
   - parallel development while Run 1501 continues;
   - **never wait for or monitor Run 1501 to epoch 5000**;
   - current task ordering;
   - deferred evaluations;
   - Run-1502 launch/continuation rules;
   - present definition of done.

If the two documents conflict, **this file takes precedence**.

Discard/ignore the obsolete intermediate file:

`HONF_Run1501_Parallel_and_Run1502_Development_Plan.md`

and any earlier `..._v2.md` copy after replacing it with this FINAL file.

In particular, any wording in the scientific reference that could be read as requiring Codex to wait for Run-1501 maturation is overridden. The epoch-5000 evaluation is a later, user-requested task.

---

## 0. Core operating rule

**Do not wait for Run 1501 to finish its 5,000-epoch continuation.**

The running Run 1501 is a separate maturation experiment. It must continue untouched.

In parallel, use the already available Run-1501 epoch-444 saved-best and epoch-500 exact checkpoints to:

1. correct evaluation/accounting issues;
2. develop an exact sparse-incidence executor without changing the learned operator;
3. implement and test one narrowly defined Run 1502 candidate.

The mature Run-1501 5,000-epoch evaluation is deferred until the user explicitly asks after training finishes.

Do not poll or monitor the long Run-1501 process.

---

## 1. Protect the running Run 1501

If the active job uses the primary checkout:

- create an ordinary temporary Git worktree from the current clean branch head;
- perform all development there;
- never edit code visible to the running process.

Use a currently free GPU only.

Do not interrupt, restart, reconfigure, inspect repeatedly, or compete with the active Run-1501 training job.

---

## 2. Current development evidence

Use as fixed development references:

- Run 1501 saved-best-total epoch 444;
- Run 1501 exact epoch 500.

These already establish that:

- query-local sparsemax routing is meaningful;
- source/query incidence reduces logical support below dense;
- saved-best fidelity is strong;
- the production reader remains rectangular;
- the current gathered pairwise selected executor is slower and higher-memory.

None of those implementation questions requires waiting for epoch 5,000.

---

## 3. Track A — repair evaluation and cost accounting

Correct the existing evidence utilities before using them for optimization decisions.

Report separately:

1. prepared P2 GPU decode;
2. full physical GPU forward;
3. application-level `predict_case` wall time.

Do not relabel `predict_case` time as pure GPU latency.

For module work use:

\[
P_{M,rect} = Q M_{pad},
\]

and report valid and padded rows separately.

Aggregate fine-row ledgers correctly across receiver chunks.

For QE distinguish logical support, actual content/geometry rows, rectangular rows, and selected rows.

Representative figures must hide padded modules, use correct physical coordinates/equal aspect, and distinguish full-grid spatial maps from Q=1024 population samples.

---

## 4. Track B — exact support-union block executor

This is execution-only. Preserve all Run-1501 mathematics.

For source type \(S\), encode positive supports:

\[
b_q = \sum_k 2^k \mathbf 1[\alpha_{qk}>0],
\qquad
b_s = \sum_k 2^k \mathbf 1[A^S_{sk}>0].
\]

A source is reachable iff:

\[
b_q \mathbin{\&} b_s 
eq 0.
\]

For every query-support signature actually observed in a receiver chunk:

1. gather queries sharing that signature;
2. form their unique supported physical-source union once;
3. gather that source bank once;
4. run the existing QM/QE calculation over the smaller rectangular block;
5. preserve each query's live rho/control/source weights;
6. scatter query outputs back once.

Do not enumerate all \(2^{12}\) masks.

Do not normalize separately per group.

Do not duplicate a physical pair through multiple groups.

Keep rectangular execution as fallback and benchmark both.

Verify output and first-gradient parity on e444/e500.

Report prepared P2 GPU time, full GPU forward, evaluator time, memory, logical support, unique pairs, and actual rows.

If this executor does not measurably help, retain the rectangular production reader and stop executor work for this task.

---

## 5. Track C — one narrow Run 1502 candidate

Run 1502 changes only the **final environmental source refinement**.

Run 1501:

\[
A^E_{j:} =
\operatorname{entmax}_{1.5}(L^E_{j:}+B^E_{j:}).
\]

Run 1502:

\[
A^E_{j:} =
\operatorname{masked\ sparsemax}(L^E_{j:}+B^E_{j:}).
\]

Keep environmental proposal entmax15.

Keep module proposal/final entmax15.

Keep query sparsemax exactly as Run 1501.

Keep K=12, D=16, geometry, prototype-anchored query keys, phase-local organization, Run-1406 fine kernels, losses, optimizer, data and seed.

No new sparsity loss, temperature, schedule, top-k, case gate or mass competition.

Implement this through one explicit sparse-incidence config option rather than copying the backend.

Historical Run-1501 checkpoints must remain unchanged.

---

## 6. Frozen Run-1502 diagnostic

Before training, use Run-1501 e444/e500 on a bounded fixed case set.

Recompute the final environment refinement with sparsemax as far as the frozen diagnostic allows.

Measure:

- environment source degree;
- environment support ratio RE;
- module support RM;
- query Kq;
- routing/output perturbation;
- empty-support pathology.

This is structural evidence, not a retrained-accuracy prediction.

Launch Run 1502 only if environment overlap falls materially without obvious support failure.

If RE barely changes, stop and report rather than launching.

---

## 7. Run 1502 workflow

If justified:

```text
run-id = 1502
run-name = sparse_incidence_environment_sparsemax
```

Use a free GPU.

Do not wait for Run-1501 epoch 5000.

- Epoch 10: numerical/training health only.
- Epoch 50: formation and learning review.
- Epoch 150: first substantive physical review.
- Epoch 500: main comparison.

Continue only while environment overlap remains improved and physical errors keep converging without persistent interface/thermal collapse.

Do not automatically continue beyond 500.

---

## 8. Deferred mature Run-1501 evaluation

When Run 1501 eventually finishes epoch 5000, **do nothing automatically**.

Wait for explicit user instruction.

At that later time, if requested:

- evaluate exact endpoint and explicit saved-best policies;
- run the 90-case mature fidelity evaluation;
- run the same sparse-incidence population analysis;
- benchmark the retained executor;
- compare mature historical references.

This is not part of the current Goal-mode task.

---

## 9. Optional scientific utilities

These may be prepared now but need not be run on the mature checkpoint yet.

### Group-conditioning intervention

Hold connectivity fixed:

\[
A^M,\ A^E,\ \alpha,\rho
\]

while neutralizing only group-dependent modulation in fine QM/QE responses.

### Design-gradient audit

Prepare utilities for output gradients, short coordinate/property sweeps, and routing-support changes.

Do not claim global smoothness from one local gradient.

---

## 10. Avoid unnecessary work

Do not:

- wait for or monitor Run 1501 epoch 5000;
- rerun full 90-case comparisons after every edit;
- launch multiple 1502 variants;
- sweep temperatures or normalizers;
- add custom kernels in this task;
- create new monitoring/provenance infrastructure;
- alter historical architectures.

Use fixed checkpoints and anchor cases during development.

---

## 11. Required report

Create:

`docs/reports/HONF_Run1501_Executor_and_Run1502_Development.md`

Separate:

1. evidence/accounting corrections;
2. executor results;
3. frozen Run-1502 diagnostic;
4. Run-1502 training evidence if launched;
5. work explicitly deferred until mature Run 1501 is requested.

State explicitly:

> Run 1501 maturation continues independently; this Goal-mode task did not wait for or monitor epoch 5000.

---

## 12. Definition of done

The current goal is complete when:

- active Run 1501 remains untouched;
- evidence/timing fixes are complete;
- support-union executor has a measured outcome;
- the Run-1502 frozen diagnostic has a clear result;
- at most one Run 1502 is launched if justified;
- no process waits for Run-1501 epoch 5000.
