# Run 1406 epoch-500 comparative evaluation

**Compared checkpoints:** exact epoch 500 of Run 1406 (`group_control_pairwise_honf`), Run 1804 (`dense_pairwise_field`), and Run 1404 (`routing_only_pairwise`).  
**Evaluation date:** 2026-09-19.  
**Scope:** predictive accuracy, synchronized GPU execution, optimizer-step cost, and learned hypergraph formation.

## Executive conclusion

Run 1406 occupies a useful middle ground, but it is not the best model on every axis.

- **Accuracy:** Run 1406 reaches pooled fluid relative L2 **0.10949**, improving on Run 1404's **0.15073** by **27.4%**, but remaining **10.9%** above Dense Run 1804's **0.09874**. It beats Run 1404 on all 90 paired cases and beats Dense 1804 on 17/90.
- **Inference:** its prepared P2 decode is **15.5% faster** than Dense 1804 (11.00 versus 13.02 ms), but its physical-state preparation is more expensive. Consequently, full forward is **10.5% slower** than Dense 1804 (36.84 versus 33.35 ms). Run 1404 is fastest at 26.33 ms.
- **Training step:** Run 1406 is about **47% faster** than Dense 1804 for both M1 and M12 canonical updates, with lower peak allocation. Run 1404 is faster still.
- **Hypergraph:** Run 1406 learns exact-zero, source-to-group incidence and nontrivial group-conditioned interaction paths. Across all 90 cases, however, mean query degree is 5.947/6, module candidate support is 100%, and environment candidate support is 99.96%. The current gain is control-state reuse and a faster prepared decoder, not material sparse elimination of fine interactions.

The practical verdict is therefore: **Run 1406 is preferable to Run 1404 when epoch-500 fidelity matters and preferable to Dense 1804 when training-step cost matters, but Dense 1804 remains the accuracy choice and Run 1404 the latency choice.** The next optimization target for 1406 is preparation overhead and genuinely selective query routing, without weakening its fidelity gains.

![Accuracy and efficiency summary](../../diagnostics/generated/interface_operator_study/run1406_epoch500_comparison_report/accuracy_efficiency_summary.png)

## 1. Experimental controls

### 1.1 Accuracy population

All three exact epoch-500 checkpoints were evaluated on the same established 90-case test/development holdout with:

- 8,192 query points per case;
- predicted ports, rather than ground-truth port substitution;
- identical case population and physical normalization;
- full-field, near/far-field, component, interface, and port metrics;
- routing maps requested only outside timed measurements.

The pooled relative L2 is computed from population-level error and target sums of squares. Equal-case summaries are also reported so large cases cannot silently dominate the conclusion.

### 1.2 Efficiency protocol

The controlled GPU benchmark used the same physical GPU 0 and anchor cases 0273 and 0653:

- Q = 8,192 and receiver chunk = 2,048;
- two warmups and five synchronized repetitions for inference;
- full forward, physical preparation plus one query, and prepared P2 decode timed separately;
- maps, profilers, and detailed routing ledgers excluded from timed calls;
- B48/Q1024 canonical optimizer updates for M1 and M12;
- one warmup and three measured updates, with checkpoint optimizer state restored;
- allocated and reserved CUDA memory measured for each phase.

Hypergraph maps were collected in separate untimed passes. Their routes are **learned interactions, not physical causality**.

## 2. Predictive accuracy

### 2.1 Whole-population field accuracy

| Exact epoch-500 checkpoint | Pooled fluid rel. L2 | Equal-case mean | Median | P95 | Maximum | Worst case |
|---|---:|---:|---:|---:|---:|---:|
| Run 1406 group control | 0.10949 | 0.10600 | 0.10621 | 0.12875 | 0.13387 | 0680 |
| Run 1804 dense | **0.09874** | **0.09611** | **0.09674** | **0.11253** | **0.12476** | 0298 |
| Run 1404 routing only | 0.15073 | 0.14543 | 0.14618 | 0.17986 | 0.20095 | 0690 |

Paired-case evidence is decisive:

| Candidate | Baseline | Candidate wins | Baseline wins | Mean L2 delta |
|---|---|---:|---:|---:|
| Run 1406 | Run 1804 | 17 | 73 | +0.00990 |
| Run 1406 | Run 1404 | **90** | 0 | -0.03943 |
| Run 1804 | Run 1404 | **90** | 0 | -0.04932 |

Thus, 1406 closes roughly four fifths of the 1404-to-1804 pooled-error gap, but it does not equal the dense model at epoch 500.

The conclusion depends on the physical domain being scored. For the all-domain metric, which includes the internal-module region, pooled/equal-case relative L2 is **0.39201/0.38495 for 1406**, 0.44891/0.44041 for 1804, and 0.45028/0.44142 for 1404. Run 1406 is therefore best on the all-domain aggregate because of stronger internal-module prediction, even though Dense 1804 is best on the fluid domain that is the primary field criterion.

### 2.2 Component and interface behavior

| Metric, pooled relative L2 | Run 1406 | Run 1804 | Run 1404 | Best |
|---|---:|---:|---:|---|
| Fluid field | 0.10949 | **0.09874** | 0.15073 | 1804 |
| Near interface | 0.12530 | **0.09759** | 0.14118 | 1804 |
| Far fluid | 0.09991 | **0.09940** | 0.16492 | 1804, near tie |
| U | 0.05552 | **0.05178** | 0.11310 | 1804 |
| V | **0.08260** | 0.10376 | 0.09328 | 1406 |
| Pressure | 0.11271 | **0.09527** | 0.13221 | 1804 |
| Vorticity | 0.15360 | **0.11367** | 0.19399 | 1804 |
| Field temperature | 0.11411 | **0.11334** | 0.19525 | 1804, near tie |
| Internal temperature | 0.07976 | **0.06949** | 0.11121 | 1804 |
| Surface temperature | 0.10362 | **0.08724** | 0.14095 | 1804 |
| Normal heat flux | 0.23782 | **0.22007** | 0.23271 | 1804 |
| Final environment temperature | 0.07400 | **0.06892** | 0.10999 | 1804 |
| Final effective h | 0.04188 | **0.04163** | 0.06268 | 1804, near tie |

Run 1406's clearest component win is V. Its largest remaining dense-baseline gaps are concentrated near the interface, in vorticity, pressure, and surface quantities. Normal heat flux is the one reported metric on which 1406 is slightly worse than both comparators.

For the visualization anchors, fluid relative L2 is 0.09263/0.07654/0.10985 on case 0273 and 0.09826/0.09239/0.12931 on case 0653 for Runs 1406/1804/1404 respectively. The anchors are consistent with, but do not substitute for, the 90-case result.

### 2.3 Training health

All three runs reached epoch 500 with finite metrics. Run 1406's validation field MSE improved through the final training segment: its best value was **0.01747 at epoch 473**, versus endpoint **0.02364**; its last-50 median was **0.02370**, improving from **0.03015** in the preceding 50 epochs. The non-monotone endpoint is ordinary validation noise rather than divergence, but the best checkpoint should remain distinct from the exact epoch-500 policy used here.

For context, exact epoch-500 endpoint validation field MSE was 0.01662 for Run 1804 and 0.03645 for Run 1404. Operational training-wall totals are not used for architecture ranking because they were not gathered under the controlled benchmark protocol.

## 3. Computation efficiency

### 3.1 Synchronized inference

Values are the mean of the two per-case medians.

| Phase | Run 1406 | Run 1804 | Run 1404 | 1406 vs 1804 |
|---|---:|---:|---:|---:|
| Full forward | 36.84 ms | 33.35 ms | **26.33 ms** | 10.5% slower |
| Physical preparation + one query | 29.46 ms | **22.96 ms** | 23.39 ms | 28.3% slower |
| Prepared P2 decode | 11.00 ms | 13.02 ms | **6.59 ms** | **15.5% faster** |

The decomposition matters: the optimized group-control contraction improves repeated prepared decoding, but preparation of its per-state control banks costs enough to reverse the result for a one-shot full forward. Workloads that reuse one prepared physical state for many query sets may therefore benefit more than one-shot inference.

Full-forward peak allocation for 1406 was 471.49/472.41 MiB on cases 0273/0653, comparable to 1804's 477.17/461.76 MiB and above 1404's 406.65/386.14 MiB. Prepared-decode peak allocation was approximately 470.14 MiB for 1406, 458.43 MiB for 1804, and 385 MiB for 1404.

Changing the reference receiver chunk to 2,048 produced only floating-order-scale differences: maximum relative output L2 was below 1.9e-6 for both 1406 and 1804, although strict elementwise `allclose` did not pass for every tensor. Run 1404 was exact in all checked tensors. These checks support timing comparability, not bitwise identity.

### 3.2 Canonical optimizer updates

| Workload | Model | Median | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|
| B48/Q1024 M1 | Run 1406 | 509.73 ms | 4,233.64 MiB | 4,520 MiB |
|  | Run 1804 | 967.73 ms | 5,847.14 MiB | 6,574 MiB |
|  | Run 1404 | **304.72 ms** | **2,700.57 MiB** | **2,958 MiB** |
| B48/Q1024 M12 | Run 1406 | 1,093.48 ms | **23,686.86 MiB** | 25,376 MiB |
|  | Run 1804 | 2,090.89 ms | 26,779.99 MiB | 28,944 MiB |
|  | Run 1404 | **617.62 ms** | 24,769.13 MiB | **24,982 MiB** |

Relative to Dense 1804, Run 1406 is **47.3% faster on M1** and **47.7% faster on M12**. Peak allocation is 27.6% lower on M1 and 11.6% lower on M12. Relative to Run 1404, however, 1406 is 67.3% slower on M1 and 77.0% slower on M12.

Model size is consistent with this middle position:

| Model | Total parameters | Trainable | Interface core | Checkpoint bytes |
|---|---:|---:|---:|---:|
| Run 1406 | 3,987,140 | 2,952,001 | 2,171,705 | 39,022,188 |
| Run 1804 | 5,430,548 | 4,395,409 | 3,615,113 | 56,380,040 |
| Run 1404 | 3,508,649 | 2,473,510 | 1,693,214 | 31,117,522 |

## 4. Hypergraph formation

### 4.1 What formed across all 90 cases

Run 1406 forms a genuine fixed-K learned control hypergraph: module and environment sources receive entmax assignments to six groups; query points route into those groups; and the fine module interaction is conditioned jointly on query, module, and group. All 90 cases completed the untimed map pass with no failures, and every P2 actual-fine-call versus unique-pair consistency check passed.

Population-level formation statistics are:

| Statistic across 90 cases | Mean | Median | Range |
|---|---:|---:|---:|
| Nonempty module groups / 6 | 2.000 | 2.000 | 2–2 |
| Nonempty environment groups / 6 | 4.389 | 4.000 | 3–6 |
| Module source degree | 1.567 | 1.600 | 1.200–2.000 |
| Environment source degree | 1.422 | 1.406 | 1.286–1.693 |
| Effective module membership groups | 1.191 | 1.181 | 1.011–1.586 |
| Effective environment membership groups | 1.168 | 1.161 | 1.085–1.291 |
| Positive query group degree / 6 | **5.947** | **6.000** | 5.609–6.000 |
| Effective query groups | 3.283 | 3.263 | 2.536–4.489 |
| Normalized query-routing entropy | 0.651 | 0.654 | 0.513–0.828 |

Logical group-path multiplicity is 1.550 for modules and 1.420 for environments when pooled by path counts. After duplicate group paths are coalesced, module support remains **100%** of valid query-source candidates and mean environment support remains **99.9567%** (minimum case 99.2020%). Thus the full population, not only the two displayed cases, establishes the lack of material execution sparsity.

The measured maps also expose learned group centres. Mean within-case pairwise centre separation is 4.197 for module centres and 4.185 for environment centres, but these are descriptive learned-coordinate distances, not physical length scales or causal influence.

### 4.2 Representative map evidence

![Run 1406 hypergraph summary](../../diagnostics/generated/interface_operator_study/run1406_epoch500_comparison_report/run1406_hypergraph_summary.png)

The two detailed epoch-500 anchors show:

| Hypergraph statistic | Case 0273 | Case 0653 |
|---|---:|---:|
| Active modules / environment sources | 3 / 192 | 5 / 192 |
| Module exact-zero fraction | 0.6667 | 0.7667 |
| Environment exact-zero fraction | 0.7457 | 0.7778 |
| Query exact-zero fraction | **0.0000** | **0.0000** |
| Mean module group degree | 2.000 | 1.400 |
| Mean environment group degree | 1.526 | 1.333 |
| Mean query group degree | **6.000** | **6.000** |
| Mean effective query groups | 3.998 | 3.309 |
| Empty module groups | 4 | 4 |
| Empty environment groups | 1 | 2 |
| Empty query groups | 0 | 0 |
| Module logical / unique paths | 49,152 / 24,576 | 57,344 / 40,960 |
| Environment logical / unique paths | 2,400,256 / 1,572,864 | 2,097,152 / 1,572,864 |
| Module / environment multiplicity | 2.000 / 1.526 | 1.400 / 1.333 |
| R_M / R_E | **1.000 / 1.000** | **1.000 / 1.000** |

The assignments are normalized to numerical precision and entmax creates many exact zeros for module and environment membership. The learned structure is also highly concentrated: only two groups carry any module source in both anchors. Environment membership uses four or five groups. Empty groups are handled without numerical failure.

### 4.3 Why this is not sparse execution yet

The important negative result is query routing: nearly every query has nonzero support on all K=6 groups. Once source incidence and query routing are joined, their union covers every valid query-module pair and almost every query-environment pair. Logical group paths have multiplicity greater than one, but the executor coalesces them to a nearly dense unique pair set. For the two displayed anchors, both actual ratios are exactly one:

\[
R_M = \frac{\text{unique module query-source pairs}}{\text{all valid module query-source pairs}} = 1,
\qquad
R_E = 1.
\]

This explains the apparently mixed efficiency result. The group-control representation enables a compact, reusable contraction and speeds the prepared decoder, but it does not yet reduce the number of unique fine interactions. Source-incidence sparsity alone is insufficient when query routing is dense.

The full 90-case evaluator agrees with the anchors: mean query degree is 5.947 of 6 while module and environment incidence remain selective. The topology is learned and differentiable, but not an execution-sparse hypergraph.

### 4.4 Comparison semantics

- **Run 1804:** hypergraph measures are *not applicable*, not zero. Its dense pairwise field has no learned source/group/query incidence topology.
- **Run 1404:** it has six legacy soft routing edges and meaningful routing statistics, but epoch-500 execution remained dense: six functional edges, zero empty selected edges, selection ratio 1.0, no gathered routes, and zero measured module/environment/query sparsity. Its query effective-edge count was approximately 3.26. These fixed-projection soft-edge statistics are not numerically interchangeable with Run 1406's entmax incidence.
- **Group identities:** group labels are permutation-ambiguous across cases. The report compares distributions, support, and costs rather than claiming group 0 in one case corresponds physically to group 0 in another.

The detailed interaction board records incidence, centres, routing, selected interaction triples, sQ/sM/sE, R_M/R_E, P0/P1/P2 ledgers, and measured prepared-decode time:

![Run 1406 learned interaction board](../../diagnostics/generated/interface_operator_study/run1406_epoch500_hypergraph/figures/group_control_interaction_board.png)

## 5. Decision and next work

Run 1406 validates the architectural idea in two respects: group-conditioned pairwise interactions recover much of 1404's fidelity deficit, and the executor produces a materially faster prepared decode and optimizer step than Dense 1804. It does not validate sparse fine execution, because R_M and R_E remain one.

Recommended next steps, in priority order:

1. Reduce physical-state preparation cost while preserving the current exact output contract.
2. Diagnose why query entmax never produces zeros even though source entmax does; focus on routing logits/scale and optimization dynamics rather than adding another loss by default.
3. Re-measure R_M/R_E and full-forward latency after any routing change; do not use logical path multiplicity as a proxy for fine-call reduction.
4. Keep Dense 1804 as the accuracy baseline and Run 1404 as the latency floor. Do not replace either based only on prepared-decode speed.

## 6. Limitations and provenance

- This is an exact epoch-500 comparison, not a best-checkpoint comparison.
- Accuracy covers all 90 established held-out development cases; this is not independent CFD verification.
- Compact Run 1406 topology statistics cover all 90 cases; full Q-by-source arrays are retained only for representative cases 0273 and 0653.
- GPU timing covers one physical GPU and two representative module workloads. Medians and memory are empirical for this protocol, not hardware-independent constants.
- Run 1404's existing mature epoch-5000 maps are intentionally excluded from the exact epoch-500 topology claims.
- Learned interaction routes must not be interpreted as recovered physical causality.

Primary machine-readable evidence:

- `diagnostics/generated/run1406_1804_1404_epoch500_90case/`
- `diagnostics/generated/interface_operator_study/epoch500_run1406_1804_1404_efficiency_20260919/`
- `diagnostics/generated/interface_operator_study/run1406_epoch500_hypergraph/`
- `diagnostics/generated/interface_operator_study/run1406_epoch500_comparison_report/summary.json`

The compact reduction is reproducible with:

```bash
PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT \
python tools/diagnostics/analyze_run1406_epoch500_comparison.py
```
