# HONF Run 1409 occupancy adaptive evaluation

**Decision:** stop the fresh candidate at epoch 50 and retain rectangular
execution. The source assignments are spatially organized, but the deployed
occupancy count collapsed to `Kplan=12` for all 90 test cases. The candidate
therefore did not learn case dependent routing capacity. It is also slower and
less accurate than the matched Run 1406 and Dense 1804 epoch 50 references.

This report describes learned routing organization. Group IDs are permutation
ambiguous, numerical assignment rank is not physical rank, and none of these
maps establishes physical causality.

## 1. Experiment identity

| Item | Executed value |
|---|---|
| Run | fresh `Run_1409_20260922_173252_occupancy_adaptive_geometric_group_control` |
| Run UUID | `d9037e86-a0f2-4642-814b-3c2ac8669492` |
| Architecture | `occupancy_adaptive_group_control_honf` |
| Initialization | fresh seed 0; neither earlier Run 1409 was resumed |
| Capacity/control | `Kmax=12`, `D=16` |
| Preparation/read | Run 1406 Dense preparation, `Cg+CM+CE`, exact rectangular fine readers |
| Port mode | predicted from epoch 1 |
| Data | established 600 train / 90 test ThermalChannel packed data |
| Training | one GPU, 50 epochs, AdamW, learning rate `3e-4`, no sweep |
| Exact review checkpoint | `epoch_0050_model.pt` |

The implemented v3 plan supersedes the earlier gate based Run 1409 proposal.
There is no always open group, hard concrete gate, optional group count loss,
stochastic gate estimator, sparsity schedule, sampled QE, or selected fine
pair kernel. The earlier `budgeted_group_control_honf` architecture and its
six open parent limit remain in the compatibility regression suite.

## 2. Implemented operator

At P0, ordinary Run 1406 controls produce unit temperature entmax 1.5
proposal assignments. Exact positive module or environment source mass defines
proposal occupancy. Source mass weighted physical centers then supply one
geometry refinement with

\[
s_{\rm geo}=0.25\sqrt{L_x^2+L_y^2}.
\]

Final exact positive occupancy defines original prototype IDs and `Kplan`.
The P0 ID plan is reused at P1 and P2; memberships, centers, collective
controls, and fine physical values are recomputed at each phase. Queries use a
learned `D=16` content score plus module and environment center distances and
are masked by both the P0 plan and current source occupancy before entmax.

The amplitude reference is

\[
\pi_k=\tfrac12(\mu^M_k+\mu^E_k),
\qquad
\kappa=\frac{1}{\sum_k\pi_k^2}.
\]

`kappa` replaces the fixed group count only at the group moment and output
scaling positions. The implementation keeps the following quantities
separate:

- registered candidate capacity: `Kmax=12`;
- deployed exact occupancy count: `Kplan`;
- packed tensor width: the maximum `Kplan` within a batch;
- effective groups: inverse concentration of each soft assignment row;
- numerical assignment matrix rank: an SVD based algebraic diagnostic;
- field amplitude: learned values after the physical readers and heads.

Logical support is computed with 16 bit masks and direct intersections. No
`2**K` table or duplicate query group source path is built. Both full width and
original ID packed forms execute the same operator.

## 3. Verification before and after training

Focused tests cover proposal and refined normalization, exact empty columns,
one step geometry arithmetic, original ID packing, closed capacity padding,
mixed case batches, source permutations, P0 plan reuse with P1/P2 refresh,
query masking of current empty sources, `kappa`, full/packed value and gradient
parity, the real predicted port path, and checkpoint state round trips.
Historical Run 1406 and gate based Run 1409 tests were run in the same pass,
including `test_six_open_budget_matches_run1406_reader`. The final combined
pass completed with **132 passed**.

The real prelaunch used two ThermalChannel predicted port batches with 24 and
48 cases and 1,024 queries per case. One in memory AdamW update produced a
finite router gradient norm of `0.0045791` and update norm of `0.039541`. All
12 prototype rows received nonzero, nonidentical gradients. The prelaunch
wrote no checkpoint and allocated no managed run.

The exact epoch 50 checkpoint passed the training resume validators for
identity, model configuration, dataset schema, normalization, optimizer group
structure, strict model state, optimizer state, and RNG state. It restored 151
parameter states containing 453 finite tensors and resolved the next epoch as
51. The audit deliberately executed no training epoch and wrote no checkpoint.

## 4. Learning through epoch 50

| Epoch | Train total | Validation total | Train field MSE | Validation field MSE | Validation temperature MSE | Validation mean group count |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.7839 | 4.3294 | 1.9379 | 1.8968 | 1.0241 | 12.000 |
| 10 | 3.3141 | 3.3053 | 1.8160 | 1.8288 | 0.7019 | 7.000 |
| 25 | 2.2591 | 2.0927 | 1.3423 | 1.2365 | 0.5083 | 11.226 |
| 50 | 0.9319 | 0.8738 | 0.4945 | 0.4541 | 0.3745 | 12.000 |

The losses learned steadily, while occupancy briefly contracted and then
returned to the full registered capacity. The epoch 50 all case audit is the
acceptance evidence; batch mean group count during training is not a population
capacity distribution.

## 5. All 90 case routing audit

### Capacity, rank, and concentration

`Kplan` is 12 for every case: mean, median, minimum, and maximum are all 12,
and every original prototype ID is occupied in all 90 cases. Active module
count ranges from 3 to 10 with mean 5.833, so `Kplan - active_modules` averages
6.167 and ranges from 2 to 9. The exact occupancy rule therefore produced
universal full capacity rather than module hub capacity or general adaptive
rank.

The soft rows remain concentrated despite full occupancy:

| Diagnostic | Mean | Range |
|---|---:|---:|
| Module effective groups per source | 1.967 | 1.427 to 2.544 |
| Environment effective groups per source | 2.610 | 2.498 to 2.744 |
| Query effective groups | 4.149 | 3.854 to 4.562 |
| Module positive degree | 3.842 | 3.000 to 4.667 |
| Environment positive degree | 4.761 | 4.562 to 5.099 |
| Query positive degree | 7.773 | 7.301 to 8.542 |
| `kappa` | 9.452 | 6.680 to 10.996 |

| Assignment matrix | Numerical rank, mean | Median | Range |
|---|---:|---:|---:|
| Module | 5.833 | 5 | 3 to 10 |
| Environment | 12.000 | 12 | 12 to 12 |
| Joint module/environment | 12.000 | 12 | 12 to 12 |
| Query | 12.000 | 12 | 12 to 12 |

The module matrix reaches the active module count, while environment, joint
source, and query matrices have full numerical rank 12 for every case. This
explains why exact occupancy remains full despite low effective group counts.
It is an algebraic SVD diagnostic and must not be read as optimal physical
rank.

### Spatial organization

Environment within group RMS radius averages 1.740 versus 3.812 under the
mass preserving shuffled baseline. Module radius averages 0.456 versus 0.956
under its shuffled baseline. Joint center separation averages 3.944 and mean
query distance to the nearest joint center is 1.175. These measurements show
spatial organization in the learned soft assignments, especially for the
environment, even though exact positive occupancy activates all 12 groups.

The group conditioned query maps also vary smoothly in space, but each query
uses 7.773 positive groups on average and 4.149 effective groups. Query routing
therefore respects the centers without producing a narrow discrete read plan.

### Logical support and true fine work

| Branch | Logical paths, mean | Unique physical pairs, mean | Support ratio, mean | Multiplicity, mean | Actual fine rows per case | Padded rows, mean |
|---|---:|---:|---:|---:|---:|---:|
| Module | 15,928 | 5,223 | 0.8668 | 3.070 | 12,288 | 789 |
| Environment | 613,318 | 170,172 | 0.8655 | 3.604 | 196,608 | 0 |

The support ratios are below one, but both branches retain roughly 87% of
dense valid query source pairs. The rectangular executor evaluates exactly
12,288 module rows and 196,608 environment rows for every audited case. The
logical path reduction therefore does not reduce actual fine work.

## 6. Full width and packed execution

Repeated deterministic evaluation gives identical `Kplan` and a maximum
prediction difference of zero. Full width and packed outputs also agree
exactly on anchors 0273 and 0653, with identical logical and actual row
ledgers. This candidate contains no stochastic gate estimator, so a
stochastic versus deterministic gate discrepancy is inapplicable.

| Case | Full width median | Packed median | Packed/full | Incremental allocated memory |
|---|---:|---:|---:|---:|
| 0273 | 86.24 ms | 95.70 ms | 1.110 | 33,744,896 / 33,744,384 bytes |
| 0653 | 79.15 ms | 86.29 ms | 1.090 | 33,744,896 / 33,744,384 bytes |

Packed execution is 9 to 11% slower and saves only 512 allocated bytes because
all cases use all 12 columns. Rectangular full width remains selected. Broad
support and the absence of a measured packed benefit do not justify enabling a
selected fine pair path.

## 7. Matched epoch 50 accuracy

The following values are equal case means over the same 90 test cases, full
grid queries, and predicted ports. Parentheses give the across case p95.

| Model | Global fluid norm L2 | Near interface norm L2 | Far fluid norm L2 | Temperature fluid norm L2 |
|---|---:|---:|---:|---:|
| Run 1409 occupancy | 0.5562 (0.6194) | 0.4326 (0.4958) | 0.5797 (0.7079) | 0.7063 (0.9169) |
| Run 1406 | 0.4189 (0.4639) | 0.3784 (0.4271) | 0.4427 (0.5196) | 0.4513 (0.5784) |
| Run 1404 | 0.6270 (0.7301) | 0.3537 (0.3958) | 0.8862 (0.9792) | 0.4772 (0.6312) |
| Dense 1804 | 0.3747 (0.4164) | 0.3302 (0.3893) | 0.3970 (0.4615) | 0.3756 (0.5098) |

Run 1409 improves on Run 1404 for global and far fluid error, but is worse near
interfaces and for the temperature channel. It is worse than Run 1406 and
Dense 1804 on all four reported field metrics.

### Physical interfaces and tails

All entries are physical relative L2, equal case mean with p95 in parentheses.

| Model | Internal T | Surface T | Normal heat flux | Final port `T_env` | Final port `h_eff` |
|---|---:|---:|---:|---:|---:|
| Run 1409 occupancy | 0.2259 (0.3364) | 0.3143 (0.4237) | 0.4637 (0.5272) | 0.3449 (0.4698) | 0.1789 (0.2777) |
| Run 1406 | 0.1899 (0.3024) | 0.2637 (0.3744) | 0.4601 (0.5197) | 0.2681 (0.3715) | 0.0785 (0.0982) |
| Run 1404 | 0.1702 (0.2564) | 0.2387 (0.3388) | 0.4597 (0.5173) | 0.2458 (0.3343) | 0.0565 (0.0713) |
| Dense 1804 | 0.1535 (0.2204) | 0.2093 (0.2918) | 0.4593 (0.5266) | 0.2842 (0.4010) | 0.0738 (0.1003) |

The candidate does not improve the physical interfaces. Its worst observed
case values are 0.3675 for internal temperature, 0.4718 for surface
temperature, 0.5812 for normal heat flux, 0.5410 for final `T_env`, and 0.3248
for final `h_eff`. The 10 module stratum has the largest mean near interface
field error, 0.4811 across 15 cases.

## 8. Matched measured cost

| Model | Mean full grid latency | p95 latency | Mean queries/s | Mean incremental peak allocated memory |
|---|---:|---:|---:|---:|
| Run 1409 occupancy | 0.5108 s | 0.5986 s | 16,224 | 113.11 MiB |
| Run 1406 | 0.3497 s | 0.3708 s | 23,456 | 122.29 MiB |
| Run 1404 | 0.0548 s | 0.0568 s | 150,360 | 557.38 MiB |
| Dense 1804 | 0.1969 s | 0.2040 s | 41,660 | 81.45 MiB |

Run 1409 is 46% slower than Run 1406 and 159% slower than Dense 1804. Its
incremental allocated memory is about 7.5% below Run 1406, but 38.9% above
Dense 1804. These are measured checkpoint and hardware results; logical
support is not substituted for execution cost.

## 9. Answers to the review questions

1. **Did a nontrivial hypergraph form?** Soft assignments and spatial centers
   are nontrivial, but the deployed occupancy plan collapsed to all 12 groups.
2. **Is `Kplan` case dependent?** No. Its histogram is `{12: 90}`.
3. **Are environment groups spatially coherent?** Yes as a learned routing
   organization: radius 1.740 versus shuffled 3.812.
4. **Are module and environment groups mutually organized?** They share
   occupied IDs and joint centers, with separated centers and concentrated
   rows. Full occupancy prevents a useful capacity claim.
5. **Does query routing respect the organization?** Spatial maps and center
   distances say yes, while a mean positive degree of 7.773 shows broad reads.
6. **Does induced query source support become sparse?** Only mildly. Mean
   support ratios are 0.8668 and 0.8655.
7. **Does execution exploit sparsity?** No. Actual rectangular row counts are
   fixed, and the packed form is slower.
8. **How does accuracy compare?** It loses to Run 1406 and Dense 1804. It beats
   Run 1404 globally and far from interfaces while losing on interface and
   physical output measures.
9. **What is the remaining limitation?** The exact positive occupancy rule is
   too permissive for adaptive capacity at epoch 50; routing remains broad,
   the executor receives no reduced work, and field learning is behind the
   strongest matched references.

## 10. Continuation recommendation

Do not continue this run to epoch 150 or 500. The required continuation gate
fails because `Kplan` is neither nontrivial nor case dependent, true fine work
does not fall, packed execution is slower, and matched accuracy regresses. No
second seed, sweep, or longer continuation was launched.

A future experiment would need a revised occupancy definition that can reject
small exact positive tails while preserving differentiable source learning.
That design change is outside this run and should not be inferred from the
current checkpoint.

## 11. Missing evidence and limits

- This is one seed and one 50 epoch candidate, as specified.
- The routing population uses a deterministic 1,024 query subset per case;
  matched accuracy and cost use the full evaluation grids.
- Numerical assignment rank uses an SVD tolerance and is not an estimate of
  optimal physical rank.
- No stochastic gate comparison exists because this architecture has no
  stochastic gate.
- No selected fine pair timing was run after broad support and packed timing
  failed the prerequisite speed test.
- No learned group label is a physical mechanism, and no causal intervention
  or new CFD solve was performed.

## 12. Evidence index

All generated evidence remains under the managed fresh run:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/
  Run_1409_20260922_173252_occupancy_adaptive_geometric_group_control/
    epoch_0050_model.pt
    metrics.csv
    run_manifest.json
    evaluations/
      prelaunch_real_update.json
      checkpoint_resume_audit.json
      full_packed_benchmark_epoch0050.json
      occupancy_population_epoch0050_q1024/
        evidence.json
        population_summary.json
        population_cases.csv
        arrays/
        figures/
      matched_epoch0050_90case/
        comparison_manifest.json
        tables/
        debug/
```

The population directory contains one compressed array package and one rendered
board per case plus the 90 case `Kplan` histogram. The matched comparison keeps
the selected case list, exact comparator checkpoints, aggregate tables, and
anchor debug arrays for cases 0273 and 0653.
