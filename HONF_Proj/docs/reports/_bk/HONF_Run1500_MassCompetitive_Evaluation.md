# Run 1500 mass-competitive adaptive hypergraph: epoch-50 evaluation

## Verdict

Run 1500 is a faithful, trainable implementation of mass-competitive group selection, but the fresh epoch-50 result does **not** form a case-dependent discrete hypergraph. Every one of the 90 test cases selects exactly two of the twelve registered prototypes, and they are always prototype IDs 0 and 5. The two surviving groups are spatially coherent and their continuous weights vary, but discrete capacity has collapsed to a fixed two-group representation.

Query weights use the two groups non-uniformly, yet positive support is almost dense: 98.57% of the audited queries have positive weight on both groups, unique module and environment support are 99.82% and 99.59% of dense, and the existing rectangular executor still evaluates 100% of registered rows. The synchronized 8,192-query full-case median is 554.37 ms, 45.4% slower than Run 1406 and 2.80 times Dense 1804. There is no execution saving.

Physical fidelity is finite and usable at this early checkpoint, but Run 1500 is worse than Run 1406 and Dense on nearly all major pooled errors. It is mixed against Run 1404 and the latest occupancy Run 1409 development context. These formation, support, cost, and fidelity results fail the continuation condition. Training stops at epoch 50; the same run was **not** resumed to 150, 500, or 5000.

![Run 1500 epoch-50 decision evidence](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/epoch50_decision_summary.png)

## Identity and protocol

- Branch: `agent/honf-core-next`
- Architecture: `mass_competitive_group_control_honf`
- Profile: `src/config_core/forward/mass_competitive_group_control_honf_context.json`
- Managed run: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph`
- Run identity: ID 1500, name `mass_competitive_adaptive_hypergraph`
- Training: fresh seed 0, AdamW, learning rate `3e-4`, predicted ports, unchanged physical losses, no sparsity loss
- Model: Kmax=12, controller width D=16, 2,057,105 parameters
- Dataset: unchanged 690-case packed ThermalChannel HDF5, existing 600/90 split
- Training device: `cuda:0` with ordinary three-GPU visibility; no `CUDA_VISIBLE_DEVICES` remapping
- Formation audit: all 90 test cases, predicted ports, deterministic 1,024-query subsets, CPU
- Accuracy/cost audit: all 90 test cases, 8,192 queries per case, predicted ports, exact epoch-50 checkpoints, synchronized `cuda:0`

The run manifest is complete with exit code 0 and last completed epoch 50. A strict read-only resume audit validates checkpoint identity, model/config state, dataset schema and normalization, finite AdamW state, RNG restoration, and next epoch 51. The audit performed no optimizer step and wrote no checkpoint.

## Implemented mathematics

The P0 router retains twelve registered candidates but lets source mass determine case support. With pre-competition source mass `pi`, the case competition is

\[
\gamma=\operatorname{sparsemax}(\log(\pi+10^{-8})),
\qquad
K_{\mathrm{case}}=\#\{k:\gamma_k>0\},
\qquad
\kappa=\frac{1}{\sum_k\gamma_k^2}.
\]

Zero-mass proposals are masked before logarithms and before the geometric refinement, so geometry cannot revive an empty proposal. P0 assignments and `gamma` remain differentiable for P1/P2; integer IDs and diagnostic support masks are detached. Final module, environment, and query assignments add `log(gamma)` before entmax. Query routing uses ordinary content, current group centres, and only positive-`gamma` groups. Full and packed execution modes gather the same P0 computations and agree in the parity test.

The implementation is opt-in. Existing Run 1406 and occupancy Run 1409 profiles retain their original factories and strict state loading.

## Prelaunch gates and learning

The frozen latest-Run-1409 prelaunch transform was deliberately diagnostic, not trained Run-1500 evidence. On its 90 recorded cases, the fixed mass-competition transform produced Kcase 2--6 (mean 3.678), with histogram `{2:9, 3:31, 4:31, 5:18, 6:1}` and kappa 1.230--4.563. It was neither universal K=1 nor K=12, so a single fresh managed run was allowed. No temperature or candidate sweep was performed.

Two real predicted-port batches (24 and 48 cases, 1,024 queries) and exactly one AdamW update completed before the managed launch. The router gradient norm was 0.01606 and parameter update norm was 0.03959; all twelve group-code rows received finite, distinct nonzero gradients. This established connectivity, not convergence.

The fresh run learned continuously:

| Epoch | Train total | Train field MSE | Validation total | Validation field MSE | Validation temperature MSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.80603 | 1.94013 | 4.35660 | 1.89774 | 1.02559 |
| 25 | 2.03505 | 1.15328 | 2.10959 | 1.25139 | 0.48346 |
| 40 | 1.10540 | 0.58811 | 0.96866 | 0.48865 | 0.34887 |
| 48 | 0.84155 | 0.44009 | **0.82226** | **0.38380** | **0.27832** |
| 50 | **0.80164** | **0.41048** | 0.84866 | 0.51289 | 0.49207 |

Epoch 48 is best-field-through-50; the required comparison uses the exact epoch-50 endpoint for every model.

## Hypergraph formation across 90 cases

### Discrete and continuous capacity

| Quantity | Result |
|---|---:|
| Registered candidates Kmax | 12 |
| Kcase histogram | `{2: 90}` |
| Kcase mean / median / range | 2.0 / 2.0 / 2--2 |
| Selected prototype frequencies | ID 0: 90; ID 5: 90; every other ID: 0 |
| Cases where Kcase equals active-module count | 0 / 90 |
| Active-module count | mean 5.833; range 3--10 |
| kappa mean / median / range | 1.9815 / 1.9938 / 1.5731--2.0000 |
| Mean pre-competition max mass | 0.5752 |
| Mean post-competition max weight | 0.5346 |
| Retained pre-competition mass | 1.0000 mean |
| Joint source assignment numerical rank | 2 in all cases |

The competition mostly balances the same two already-positive groups: mean `max(pi)` is 0.575, while mean `max(gamma)` is 0.535. Because retained pre-competition mass is 1.0, sparsemax competition does not eliminate additional positive `pi` mass at epoch 50; the earlier source assignments have already zeroed the other ten candidates. This is a fixed two-group collapse, not module-hub behavior and not broader case-adaptive grouping.

![All 90 cases select Kcase 2](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/figures/kplan_histogram.png)

### Source organization

The two groups are geometrically meaningful as learned partitions, but learned labels remain permutation-ambiguous and are not causal physical mechanisms.

| Source statistic | Module | Environment |
|---|---:|---:|
| Positive source degree, mean | 1.350 | 1.183 |
| Effective groups per source, mean | 1.117 | 1.052 |
| Normalized entropy, mean | 0.046 | 0.021 |
| Assignment numerical rank, mean | 1.989 | 2.000 |
| Learned RMS radius, mean | 1.418 | 2.465 |
| Mass-preserving shuffled RMS radius, mean | 2.068 | 3.849 |
| Learned radius reduction versus shuffled | 31.4% | 35.9% |

Mean joint-centre separation is 5.147 (range 3.703--6.458). The surviving groups are therefore spatially coherent relative to the mass-preserving shuffle. However, the occupancy Run 1409 context had stronger relative compactness reductions: 52.3% for modules and 54.4% for environment. Its K=12 organization is not directly radius-matched to this K=2 result, but the shared shuffled normalization shows no clustering improvement from mass competition.

![Environment compactness relative to shuffled mass](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/figures/environment_compactness_vs_shuffled.png)

### Query organization and physical support

The dominant query maps are spatially divided, but positive support is not selective enough to reduce pair support:

- mean query degree is 1.986 of Kcase=2;
- 90,839 of 92,160 sampled queries (98.57%) use both groups, and 1,321 use one;
- mean query effective groups is 1.314 and mean normalized entropy is 0.135;
- mean query-to-joint-centre distance is 2.355;
- mean module logical paths / unique pairs are 8,017 / 5,966, multiplicity 1.348;
- mean environment logical paths / unique pairs are 231,102 / 195,803, multiplicity 1.180;
- unique support ratios are `R_M^support=0.99823` and `R_E^support=0.99591`.

For every 1,024-query case audit, the executor reports 12,288 module rows and 196,608 environment rows: exactly `Q*Kmax` and `Q*192`. Logical support, unique pairs, and executed rows are distinct quantities here. The two-group representation does not yield adaptive computation.

Representative cases 0273 and 0653 show modules, environment tokens, `pi`, `gamma`, joint centres, dominant query group, query degree, and one query-to-groups-to-source-support diagram:

![Run 1500 case 0273 mass-competitive organization](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/figures/mass_competitive_board__0273.png)

![Run 1500 case 0653 mass-competitive organization](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/figures/mass_competitive_board__0653.png)

## Matched physical fidelity

The table reports pooled relative L2 on the same 90 test cases, exact epoch-50 endpoints, predicted ports, and 8,192 queries per case. Run 1409 is development context from its already-completed matched evaluation; it was not rerun as a fifth model in the new four-checkpoint artifact.

| Quantity | Run 1500 | Run 1406 | Run 1404 | Dense 1804 | Occupancy 1409 context |
|---|---:|---:|---:|---:|---:|
| Fluid field | 0.63058 | 0.42440 | 0.61834 | **0.37203** | 0.56102 |
| Near interface | 0.42465 | 0.38579 | 0.35771 | **0.33349** | 0.44477 |
| Far fluid | 0.72698 | 0.44281 | 0.88683 | **0.40406** | 0.58152 |
| Vorticity | 0.60549 | **0.51200** | 0.71607 | 0.52940 | 0.52055 |
| Field temperature | 0.81748 | 0.45634 | 0.50753 | **0.36337** | 0.67216 |
| Internal temperature | 0.23781 | 0.22218 | 0.18659 | **0.16537** | 0.24814 |
| Surface temperature | 0.31708 | 0.29261 | 0.24990 | **0.21832** | 0.33043 |
| Normal heat flux | **0.47201** | 0.47405 | 0.47580 | 0.47942 | 0.47723 |
| Final outside temperature | 0.35200 | 0.29246 | **0.25581** | 0.26667 | 0.35102 |
| Final effective h | 0.10093 | 0.08069 | **0.05879** | 0.07824 | 0.17151 |

Run 1500 only wins the small heat-flux difference among these exact-50 models. It is worse than Run 1406 and Dense on every other listed quantity. Against Run 1404 it improves far-fluid, vorticity, and heat flux, but regresses fluid, near-interface, both temperature measures, outside temperature, and effective h. It is better than occupancy Run 1409 on near-interface, internal/surface temperature, heat flux, and effective h, but worse on the main fluid, far-fluid, vorticity, and field-temperature measures. The candidate is physically usable, but it does not meet the preferred Run-1404-to-Run-1406 fidelity band on the major global/interface metrics.

These are model-versus-dataset metrics. They are not new CFD validation or evidence that learned groups are physical causes.

## Measured cost

Each row uses the same synchronized full-case evaluator with 8,192 queries, predicted ports, routing maps, and the same two saved anchor maps.

| Model | Median latency | Mean incremental peak CUDA memory |
|---|---:|---:|
| Run 1500 | 554.37 ms | 125.37 MiB |
| Run 1406 | 381.32 ms | 121.67 MiB |
| Run 1404 | 55.11 ms | 557.38 MiB |
| Dense 1804 | 197.99 ms | **81.45 MiB** |
| Occupancy 1409 context | 483.99 ms | 113.11 MiB |

Run 1500 is 45.4% slower than Run 1406, 14.5% slower than occupancy Run 1409, and 2.80 times Dense 1804. Training at batch size 48 and 1,024 queries recorded a peak allocated CUDA memory of 24,216 MiB. Since the executor evaluates the complete registered rectangles, the measured regression is consistent with zero execution sparsity. A selected-execution benchmark is not warranted by this checkpoint.

## Answers to the nine required questions

1. **Did mass competition produce a nontrivial case-dependent K?** No. It produced exact sparse support, but universal Kcase=2 with the same two prototypes in all 90 cases.
2. **Is K related only to module count or also to environment/operating complexity?** Neither at epoch 50: K is constant while module count varies from 3 to 10. No discrete dependence on module, environment, or operating complexity can be inferred. Only continuous `gamma` and `kappa` vary.
3. **Are surviving hyperedges spatially coherent?** Yes as learned geometry: source radii are 31.4% and 35.9% below mass-preserving shuffled references, and dominant query maps form spatial regions. This is learned organization, not physical causality.
4. **Does mass competition improve or damage source clustering?** The two surviving groups are coherent, but relative compactness is weaker than occupancy Run 1409's shuffled-normalized context. There is no evidence of an improvement; the reduction to two broad regions coarsens the organization.
5. **Does query routing use the case hypergraph selectively?** It uses the two groups with varying weights and spatially varying dominance, but not selectively in positive support: 98.57% of sampled queries use both.
6. **Does query-source support become materially sparse?** No. Unique module/environment support remains 99.82%/99.59% of dense.
7. **Does support reduction translate into measured execution savings?** No. Actual rows are 100% of both registered rectangles, latency regresses, and memory does not improve.
8. **How does physical fidelity compare with 1404, 1406, and Dense?** It is broadly worse than Run 1406 and Dense, and mixed but mostly worse than Run 1404. The only listed aggregate win is a small heat-flux improvement.
9. **Does the evidence support discrete dynamic K, or only continuous effective complexity kappa?** It does not support discrete dynamic K. It supports a fixed two-group representation with continuous weighting; even that continuous complexity is concentrated near two (`kappa` mean 1.982, median 1.994), with a limited tail down to 1.573.

## Continuation decision

The design requires nontrivial Kcase, coherent geometry, healthy learning, and no severe physical collapse before continuing the same run. Learning and geometry are healthy, and the model remains finite, but discrete K is not case-dependent, query-source support is nearly dense, actual work is fully dense, cost regresses, and major fidelity is weak. The evidence therefore supports **stop at exact epoch 50**. No epoch-150/500 resume and no 5000-epoch extension were launched.

## Verification

- Frozen Run-1409 prelaunch diagnostic: 90 cases, fixed transform, no managed run or sweep.
- Real predicted-port integration: two batches plus exactly one AdamW update with finite router gradients and updates.
- Fresh managed Run 1500: 50/50 epochs complete, exit code 0, endpoint checkpoint present.
- Strict checkpoint resume audit: complete; no training epoch executed and no checkpoint written.
- Focused architecture, config, registry, full/packed parity, historical strict-load, and evidence tests pass.
- Historical Run-1406/1409 compatibility and ThermalChannel integration tests pass.
- `py_compile`, JSON parsing, and `git diff --check` pass.
- Ruff was not available in the ModularDT environment and is not claimed.

## Structured evidence

- Run manifest: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/run_manifest.json`
- Training metrics: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/metrics.csv`
- Resume audit: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/checkpoint_resume_audit.json`
- 90-case formation evidence: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/evidence.json`
- Formation cases: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/mass_competitive_population_epoch0050_q1024/population_cases.csv`
- Matched fidelity/cost tables: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1500_20260922_193820_mass_competitive_adaptive_hypergraph/evaluations/matched_epoch0050_90case/tables/`
- Frozen prelaunch diagnostic: `diagnostics/generated/run1500_mass_competition_prelaunch/summary.json`
- Real-update evidence: `diagnostics/generated/run1500_mass_competition_prelaunch/real_update.json`

## Limits

- This is a single-seed, exact-epoch-50 experiment. Stopping is based on the specified formation gate, not proof that no later checkpoint could ever change support.
- Population routing maps use deterministic 1,024-query subsets; physical fidelity and synchronized cost use all 8,192 queries per case.
- Learned group labels are permutation-ambiguous, and routing support does not identify physical causal mechanisms.
- Logical paths, unique pairs, actual executed rows, memory, and latency are reported separately. A lower Kcase is not itself computational sparsity.
- No new CFD truth, mesh-independence study, out-of-distribution geometry study, hyperparameter sweep, temperature tuning, or custom selected kernel was run.
