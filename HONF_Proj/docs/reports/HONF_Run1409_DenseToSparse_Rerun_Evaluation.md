# HONF Run 1409 dense-to-sparse rerun evaluation

**Status:** completed through the planned epoch-150 hardening point; stopped
there. The candidate learned useful field and interface predictions, but it
did not learn case-specific capacity. Its deployed gate plan remained fully
open and its reader performed rectangular fine work. This is evidence about
one seed and one calibrated continuation, not a claim about optimal physical
rank, physical causality, or general fine-pair sparsity.

## 1. Candidate identity

| Item | Executed identity |
|---|---|
| Architecture | `budgeted_group_control_honf` |
| Capacity / control width | `Kmax=12`, `D=16` |
| Preparation | Run-1406 Dense MM/ME/EM preparation |
| Physical control | `Cg+CM+CE` |
| Query readers | Run-1406 QM/QE paths; no duplicate query-group-source reader and no sampled QE |
| Core profile | `src/config_core/forward/budgeted_group_control_honf_context.json` |
| Experiment overlay | `src/config_core/forward/experiments/run1409_case_group_budget_calibrated.json` |
| Managed run | `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1409_20260922_150032_case_budgeted_group_control_dense_to_sparse_rerun` |
| Dataset / split | `thermal_channel_global_v1`, 600 train / 90 test cases |
| Seed / device | seed 0, one free `cuda:0` |
| Parameters | 2,059,250 |
| Checkpoints used | `epoch_0050_model.pt`, `epoch_0100_model.pt`, and completed endpoint `latest_model.pt` at epoch 150 |
| Port condition | predicted |

The run manifest is `completed` with `last_completed_epoch=150`. Historical
checkpoints and the earlier Run-1409 report were retained. No baseline copies,
hash gates, approval infrastructure, custom kernels, or monitoring service were
created.

## 2. Implemented operator and continuation

All 12 groups have symmetric case-conditioned hard-concrete gates. There is
no privileged always-open group. If all raw deterministic gates close at full
hardening, the group with the largest logit is selected as a one-group fallback.
The P0 gate plan is sampled once and reused through P1/P2. Memberships,
collective controls, and fine physical values are recomputed at P1/P2.

For epoch `t`, the routing-logit scale is

\[
r(t)=0.1+0.9\min\left(\frac{t-1}{24},1\right),
\]

and the gate continuation is

\[
c(t)=\begin{cases}
0,&t\le 25,\\
(t-25)/125,&26\le t\le150,\\
1,&t\ge150,
\end{cases}
\qquad
z_{\mathrm{eff}}=(1-c)+c\tilde z.
\]

The same `z_eff` is applied before module-source, environment-source, and query
entmax normalization. The gate-only reference distribution and amplitude
factor are

\[
\eta=\operatorname{entmax}_{1.5}(\log z_{\mathrm{eff}}),
\qquad
\kappa=\left(\sum_k\eta_k^2\right)^{-1}.
\]

`Kmax`, `Kraw`, `Kexecuted`, `Klive`, `Kpack`, and `kappa` are logged
separately. Neither a packed width nor a routing rank is interpreted as field
amplitude or actual fine work.

The only added training term is applied once per case/forward:

\[
L_{K}=\lambda(t)\,\operatorname{ReLU}\left(\sum_{k=1}^{12}p_k-1\right),
\qquad
\lambda(t)=c(t)\times0.005783974924700852.
\]

The coefficient was calibrated once from training data. There was no sweep,
second seed, auxiliary routing penalty, extra schedule, or automatic
continuation.

Two implementations execute the same operator: a full-width path and a
compact-column path with original prototype IDs and explicit padding masks.
Selected fine reads were not enabled because the measured compact path did not
reduce actual fine rows, memory, or latency consistently.

## 3. Verification before launch

### Focused algebra and integration coverage

The focused tests cover:

- the six-forced-open limit against the Run-1406 group-control reader;
- closed-padding invariance when registered capacity changes from 12 to 32;
- mixed-case full-width/compact prediction and gradient parity;
- the all-closed highest-logit fallback;
- phase-shared P0 gating with refreshed P1/P2 controls and fine values;
- availability before source/query entmax normalization;
- the exact `r(t)`, `c(t)`, and `lambda(t)` boundaries;
- the single expected optional-group-count objective;
- strict checkpoint loading, optimizer slots, progress state, RNG restoration,
  and the next stochastic gate draw;
- one real predicted-port optimizer update.

The final complete focused suite result is recorded in Section 11.

### Real-data epoch-1 prelaunch audit

`evaluations/prelaunch_epoch1_audit.json` executed exactly two real
`B=48, Q=1024` predicted-port batches and one optimizer step on GPU. It did
not allocate a managed run or write a checkpoint.

| Check | Result |
|---|---|
| Protocol | pass |
| Schedule | `r=0.1`, `c=0`, effective budget weight 0 |
| Raw counts | batch 1: Kraw 9–12; batch 2: Kraw 10–12 |
| Executed counts | 12 for all 96 cases because `z_eff=1` at `c=0` |
| `eta`, `kappa` | `eta=1/12`; `kappa≈12` |
| Routing entropy | module/environment/query near-uniform fraction 1.0 in both batches |
| Prototype gradients | all 12 rows finite, nonzero, and non-identical in both batches |
| Gate-network gradient | zero as expected at `c=0` |
| Phase behavior | P0 plan reused; P1/P2 fine sources refreshed |
| Optimizer | one finite, nonzero step; update norm 0.463381 |

This audit verifies data flow and gradients at initialization. It is not
evidence of learned sparsity.

## 4. Training and resume review

Training ran fresh to epoch 50, was reviewed, and resumed the same managed run
from `latest_model.pt`. The resume log continued at epoch 51 with the same
architecture and optimizer inventory. The test suite independently verifies
strict parameter, optimizer, schedule-progress, and RNG restoration.

| Epoch | `c(t)` | Validation total | Validation field MSE | Validation temperature MSE |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 4.40850 | 1.90199 | 1.07311 |
| 25 | 0.000 | 2.12238 | 1.25566 | 0.47918 |
| 50 | 0.200 | 0.648219 | 0.357497 | 0.20429 |
| 100 | 0.600 | 0.264317 | 0.116069 | 0.05430 |
| 150 | 1.000 | 0.240137 | 0.103969 | 0.033592 |

Best validation total was 0.198891 at epoch 147, best field MSE was 0.067968
at epoch 146, and best temperature MSE was 0.028143 at epoch 145. Accuracy
learning was healthy enough to justify the bounded continuation from 50 to
150. The endpoint regressed from those best values, so the relevant best
checkpoint should be chosen according to the intended validation objective.

## 5. Capacity, routing rank, and actual work

The deterministic population audit uses all 90 held-out cases with Q=1024.
Ranks are weighted learned routing ranks from thin QR, not physical operator
ranks or optimal ranks.

| Quantity | Epoch 50 | Epoch 150 |
|---|---:|---:|
| `Kmax` | 12 | 12 |
| `Kraw=Kexecuted=Klive=Kpack=12` | 90/90 | 90/90 |
| Mean `Kexpected` | 11.997406 | 11.999977 |
| `Kexpected` range | 11.994975–11.998216 | 11.999928–11.999994 |
| Mean optional expected count | 10.997406 | 10.999977 |
| Fallback used | 0/90 | 0/90 |
| Query reached groups | 12 in 90/90 | 12 in 90/90 |
| Module nonempty groups | 1:84, 2:6 | 1:89, 2:1 |
| Module thin-QR rank | 1:84, 2:6 | 1:89, 2:1 |
| Environment nonempty groups | 7:53, 8:22, 9:15 | 6:23, 7:24, 8:39, 9:4 |
| Environment thin-QR rank | 7:55, 8:30, 9:5 | 6:38, 7:21, 8:31 |
| Mean environment entropy rank | 1.31395 | 1.30926 |

The environment incidence uses fewer nonempty/rank directions than registered
capacity, and the module path is effectively rank one for most cases. This
organization did not translate into gate capacity reduction.

The executor ledger reports `rectangular_reference` for every case:

| Phase | Module actual fine rows | Environment actual fine rows | Module padded rows | Environment padded rows |
|---|---:|---:|---:|---:|
| P0 / P1 | 9,216 | 147,456 | 1,536–6,912 | 0 |
| P2 | 12,288 | 196,608 | 2,048–9,216 | 0 |

Module logical/unique work varies with the case: 2,304–7,680 rows at P0/P1
and 3,072–10,240 rows at P2. Environment unique and actual rows stay fully
rectangular. Logical paths, unique pairs, padding, and actual calls therefore
must remain separate; lower learned rank did not reduce executed fine work.

## 6. Full-width versus compact execution

At epoch 150, predictions agreed exactly (`absolute_max=0`) between the two
implementations on both anchors. Median CUDA timings are full-width / compact:

| Case | Full forward | Preparation + query | P2 decode |
|---|---:|---:|---:|
| 0273 | 32.567 / 33.997 ms | 32.105 / 33.837 ms | 2.823 / 2.832 ms |
| 0653 | 32.188 / 32.734 ms | 36.482 / 31.834 ms | 2.911 / 2.830 ms |

Peak allocated memory was identical between modes: 244.007 MiB for the full
forward, 188.570 MiB for preparation/query, and 241.489 MiB for P2 decode.
Peak reserved memory was 424 MiB. Timing changes are small and inconsistent,
and actual fine rows are identical, so compact columns are an algebraically
equivalent implementation rather than an execution win at this checkpoint.

## 7. Deterministic and stochastic gate audit

Eight explicit stochastic P0 draws were run for each anchor and execution
mode at epochs 50 and 150. The same sampled plan was reused through later
phases in each forward.

| Checkpoint | Anchors / modes | Deterministic count | Eight stochastic counts | Prediction disagreement |
|---|---|---:|---:|---:|
| Epoch 50 | 0273, 0653 / full, compact | 12 | all 12 | 0 |
| Epoch 150 | 0273, 0653 / full, compact | 12 | all 12 | 0 |

At epoch 150 raw and effective stochastic gates were all one and
`kappa≈12`. The audit therefore finds exact stochastic/deterministic agreement
because the learned policy saturated open; it does not establish robustness
for a genuinely sparse stochastic plan.

## 8. Matched accuracy comparisons

### Epoch 50 decision point

The exact epoch-50 comparison used the same 90 cases, predicted ports, and
full-grid evaluator for Run 1409 v2, the earlier Run 1409 design, Run 1406,
Run 1404, and Dense Run 1804.

| Model | Pooled fluid relative L2 | Pooled fluid MSE | Equal-case median / P95 |
|---|---:|---:|---:|
| Run 1409 v2 | 0.51123 | 0.24196 | 0.49804 / 0.58756 |
| Run 1409 v1 | 0.54911 | 0.27915 | 0.54832 / 0.62319 |
| Run 1406 | 0.42440 | 0.16675 | 0.42068 / 0.46392 |
| Run 1404 | 0.61834 | 0.35397 | 0.62210 / 0.73007 |
| Dense Run 1804 | 0.37203 | 0.12813 | 0.37449 / 0.41637 |

V2 beat v1 on global fluid L2 in 73/90 cases, but beat Run 1406 in 0/90
and Dense 1804 in 2/90. The stable learning curve and only partial
continuation (`c=0.2`) supported continuing the same run to hardening.

### Exact epoch-100 matched comparison

Epoch 100 is the latest exact checkpoint shared by this candidate and all
three requested historical baselines.

| Metric | Run 1409 v2 | Run 1406 | Run 1404 | Dense Run 1804 |
|---|---:|---:|---:|---:|
| Pooled fluid relative L2 | 0.23916 | 0.25645 | 0.46987 | 0.22383 |
| Pooled fluid MSE | 0.05295 | 0.06089 | 0.20439 | 0.04638 |
| Equal-case fluid L2 mean / P95 | 0.2312 / 0.2914 | 0.2530 / 0.2859 | 0.4780 / 0.6138 | 0.2213 / 0.2460 |
| Near-interface L2 mean / P95 | 0.2598 / 0.3115 | 0.2707 / 0.3179 | 0.2570 / 0.2931 | 0.2061 / 0.2599 |
| Far-fluid L2 mean / P95 | 0.2109 / 0.2927 | 0.2474 / 0.2883 | 0.6876 / 0.8473 | 0.2146 / 0.2485 |
| Fluid-temperature MAE mean / P95 | 1.0108 / 1.6552 | 1.2843 / 1.7888 | 1.1682 / 1.7601 | 1.0018 / 1.3525 |
| Internal-temperature MAE mean / P95 | 1.0401 / 1.8991 | 1.2092 / 1.9650 | 1.2471 / 2.1449 | 0.7825 / 1.5863 |
| Surface-temperature MAE mean / P95 | 1.1696 / 2.1335 | 1.3697 / 2.1220 | 1.4060 / 2.2870 | 0.9514 / 1.6952 |
| Normal heat-flux MAE mean / P95 | 4.5989 / 6.3708 | 4.8187 / 6.3823 | 4.4707 / 5.9646 | 4.8075 / 6.4566 |

Run 1409 v2 beat Run 1406 on global fluid L2 in 76/90 cases and Run 1404
in 90/90. It beat Dense 1804 in 29/90. It improved Run 1406 on fluid
temperature MAE in 87/90 and normal heat-flux MAE in 85/90, while its port
environment-temperature MAE beat Run 1406 in only 7/90. The result is a
useful accuracy tradeoff, not evidence of budgeted execution.

### Epoch-150 candidate endpoint

There is no exact epoch-150 checkpoint for the historical baselines, so the
completed candidate endpoint is reported without fabricating a matched
baseline row.

| Metric | Run 1409 v2 epoch 150 |
|---|---:|
| Pooled fluid relative L2 / MSE | 0.25087 / 0.05826 |
| Equal-case fluid L2 mean / P95 | 0.2477 / 0.2835 |
| Near-interface L2 mean / P95 | 0.2243 / 0.2839 |
| Far-fluid L2 mean / P95 | 0.2590 / 0.2965 |
| Pressure relative L2 | 0.20928 |
| Vorticity relative L2 | 0.36334 |
| Fluid-temperature MAE mean / P95 | 0.8969 / 1.2266 |
| Internal-temperature MAE mean / P95 | 0.8100 / 1.4174 |
| Surface-temperature MAE mean / P95 | 1.0309 / 1.6478 |
| Normal heat-flux MAE mean / P95 | 3.8028 / 5.1062 |
| Pressure-drop absolute error mean / P95 | 0.01266 / 0.03029 |
| Outlet-temperature absolute error mean / P95 | 1.0095 / 2.2099 |
| Active-module-temperature absolute error mean / P95 | 0.3156 / 0.8230 |

The endpoint has strong interface and temperature accuracy, while pooled
fluid relative L2 is slightly worse than its exact epoch-100 value.

## 9. Measured evaluation cost

The matched evaluator measured one full-grid case at a time on the same GPU.
These figures include each model's existing executed policy; Run 1406 remains
its ordinary rectangular timed path, distinct from diagnostic support maps.

| Exact epoch 100 model | Mean / P95 wall time | Mean peak allocated | Mean incremental peak |
|---|---:|---:|---:|
| Run 1409 v2 | 0.2500 / 0.2554 s | 66.57 MiB | 43.29 MiB |
| Run 1406 | 0.2275 / 0.2328 s | 66.72 MiB | 43.11 MiB |
| Run 1404 | 0.0541 / 0.0558 s | 386.32 MiB | 364.27 MiB |
| Dense Run 1804 | 0.1952 / 0.2023 s | 82.96 MiB | 53.58 MiB |

Run 1409 v2 is about 9.9% slower than Run 1406 by these means, with similar
incremental allocation. Its epoch-150 candidate-only mean / P95 was
0.2522 / 0.2534 s with the same 66.57 MiB mean peak allocation. Since the
gates remain open and the backend executes rectangular rows, no capacity
speedup is present.

## 10. Missing evidence and limits

- Epoch 150 is stored as the completed `latest_model.pt`; the maintained
  milestone policy did not create an `epoch_0150_model.pt` copy.
- Exact historical comparison is available at epoch 100. No exact epoch-150
  Run-1406/1404/1804 checkpoints exist, so no unmatched comparison is shown.
- No epoch-500 candidate evidence exists because continuation was not
  scientifically warranted.
- Selected fine reads were not enabled. Full and compact rectangular paths
  had identical fine rows and memory, with no consistent latency win.
- The stochastic audit probes two anchors with eight draws each. It found
  saturation, so it cannot characterize a nontrivial sparse stochastic policy.
- Learned routing ranks and group labels are permutation-ambiguous and do not
  identify physical causal rank.

## 11. Recommendation and final decision

**Stop this candidate at epoch 150. Do not continue it to epoch 500.**

The bounded continuation answered the scientific question. Accuracy became
competitive with Run 1406 by epoch 100 and several physical interface metrics
continued to improve. Capacity moved in the wrong direction: mean expected
count increased from 11.9974 at epoch 50 to 11.99998 at full hardening, all 90
cases retained 12 raw and executed groups, stochastic draws also retained 12,
and true fine work stayed rectangular. Validation objectives peaked at
epochs 145–147 and regressed by the endpoint. Another 350 epochs would test
longer dense accuracy optimization rather than the proposed case-budgeted
capacity mechanism.

Final verification before delivery:

- focused algebra, routing, integration, checkpoint, and workflow tests:
  **165 passed**;
- `py_compile`: passed for every changed Python source and diagnostic;
- `git diff --check`: passed.

## 12. Evidence index

All generated evidence is under the managed run's `evaluations/` directory:

- `prelaunch_epoch1_audit.json`
- `population_capacity_rank_epoch0050_q1024.json` and `.csv`
- `population_capacity_rank_epoch0150_q1024.json` and `.csv`
- `anchor_evidence_epoch0050.json`
- `anchor_evidence_epoch0150.json`
- `matched_epoch0050_90case/`
- `matched_epoch0100_90case/`
- `candidate_epoch0150_90case/`

Historical context remains in
`docs/reports/HONF_Case_Budgeted_Dynamic_K_Evaluation.md` and
`docs/reports/HONF_Run1406_Epoch500_Comparative_Evaluation.md`.
