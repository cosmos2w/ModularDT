# Run 1408 hypergraph quadrature: implementation and epoch-500 evaluation

## Verdict

Run 1408 is a faithful opt-in test of the proposed environmental quadrature idea, but it is **not a successful replacement for the mature executors at epoch 500**.

The implementation changes only fine environmental receiver integration: six shared many-body controllers emit four query-conditioned sample sites and positive within-group weights, current-phase fine environmental keys, values, and scalar controls are bilinearly interpolated at those 24 sites, and content/geometry attention uses the design's mass-weighted numerator and denominator while retaining the parent branch amplitude. Dense MM/ME/EM preparation, the phase-shared P0 controller, K=6/D=16, module reading, `C_g+C_M+C_E`, P0/P1/P2 and local-surrogate coupling, losses, optimizer, and data remain inherited.

The fresh run learned continuously and did not collapse. Its sample locations and within-group weights matter to the frozen model. However, the exact epoch-500 candidate is worse than Run 1406, Run 1407, and Dense 1804 on the main 90-case fluid, near-interface, vorticity, temperature, and heat-flux measures. More importantly, the theoretical reduction from 192 environmental tokens to 24 sample sites did **not** produce a measured speedup: synchronized full and prepared-P2 latency are slower than every mature comparator. Run 1408 therefore stops at 500 epochs. There was no sweep, second candidate, or 5000-epoch continuation.

## Identity and scope

- Repository branch: `agent/honf-core-next`
- Design reference: `07a255ea0acfd7e61d46aa58324baf6535242b2b`
- Architecture/profile: `hypergraph_quadrature_honf`
- Successful run: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1408_20260920_133744_hypergraph_quadrature`
- Dataset: the unchanged 690-case packed ThermalChannel HDF5, with the existing 600/90 split
- Successful training source: `91bafb9c1d7510b82f0068d8f9900cacb5a1f0b3`; the later `d8be2cd` changes only evidence interventions/tests
- Device: free GPU 2, with ordinary device visibility and `--device cuda:2`
- Checkpoint policy: fresh training to 50, scientific review, then the **same run and optimizer state** resumed to 500

An earlier directory, `Run_1408_20260920_133604_hypergraph_quadrature`, failed before epoch 1 because predicted port receivers can lie outside the environmental interpolation bounds. The correction clamps only the receiver reference used to construct convex sample sites; generic interpolation remains strict about bounds. That failed launch has no checkpoint and is not part of the learning result.

This candidate does not add a tree, direct group-value branch, top-k selector, balance or sparsity losses, dense QE fallback, or custom kernel. It is a quadrature approximation, not an exact Run 1407 executor rewrite. Generic regular-grid sampling is isolated in `environment_sampling.py`; the ThermalChannel adapter owns bounds, coordinates, quadrature weights, and token-to-grid mapping.

## Learning and checkpoint policy

Run 1408 improved materially after its early lag, which justified continuing the same run from 50 to 500. All logged core losses remained finite.

| Epoch | Run 1408 validation field MSE | Run 1407 | Run 1406 |
|---:|---:|---:|---:|
| 50 | 0.65376 | 1.32424 | 0.30036 |
| 100 | 0.34074 | 0.15433 | 0.12627 |
| 250 | 0.14765 | 0.09341 | 0.06015 |
| 500 | 0.06712 | 0.01861 | 0.02364 |

At epoch 500, Run 1408 has training total loss 0.10384, training field MSE 0.06053, validation total loss 0.11884, validation field MSE 0.06712, and validation temperature MSE 0.01723. The saved best-field-through-500 checkpoint is epoch 500, so the endpoint and field-selection policies coincide for the primary 90-case evaluation. Best total validation loss is 0.11294 at epoch 462; best validation temperature MSE is 0.01652 at epoch 498. Mature Run 1406/1407 best-through-500 weights were not available as a matched policy, so physical comparisons use exact epoch 500 for every model.

## Physical fidelity

The table reports pooled relative L2 across the same 90 test cases with predicted ports and 8,192 query points. Parent and Dense checkpoints are also exact epoch 500. P95 and maximum columns are equal-case tail statistics for Run 1408.

| Quantity | Run 1408 | Run 1406 | Run 1407 | Dense 1804 | Run 1408 P95 | Run 1408 max |
|---|---:|---:|---:|---:|---:|---:|
| Fluid field | 0.16893 | 0.10949 | 0.10103 | **0.09874** | 0.21290 | 0.23719 |
| Near interface | 0.20692 | 0.12530 | 0.10816 | **0.09759** | 0.25886 | 0.28543 |
| Far fluid | 0.10956 | 0.09991 | 0.09994 | **0.09940** | 0.13298 | 0.16092 |
| Vorticity | 0.23888 | 0.15360 | 0.13678 | **0.11367** | 0.33350 | 0.36277 |
| Field temperature | 0.12603 | 0.11411 | **0.11015** | 0.11334 | 0.15562 | 0.22254 |
| Internal temperature | 0.07165 | 0.07976 | **0.05897** | 0.06949 | 0.09098 | 0.10757 |
| Surface temperature | 0.09428 | 0.10362 | **0.08025** | 0.08724 | 0.11764 | 0.13951 |
| Normal heat flux | 0.24051 | 0.23782 | 0.23508 | **0.22007** | 0.27393 | 0.28974 |
| Final outside temperature | 0.08739 | 0.07400 | **0.06833** | 0.06892 | 0.14733 | 0.24838 |
| Final effective h | 0.06573 | 0.04188 | **0.03340** | 0.04163 | 0.07302 | 0.07632 |

Run 1408 loses to every exact-500 comparator on fluid, near-interface, and vorticity error in all 90 paired cases. It beats Run 1406 on pooled internal and surface temperature, but not Run 1407. Against Dense it wins 21/90 far-fluid cases, 20/90 field-temperature cases, and 39/90 internal- and surface-temperature cases; those isolated wins do not offset the aggregate regressions. The worst fluid case is 0682 (0.23719), and the same case has the worst near-interface error (0.28543).

For historical context, the maintained exact-500 Run 1404 artifact reports fluid 0.15073, near-interface 0.14118, far-fluid 0.16492, vorticity 0.19399, field temperature 0.19525, heat flux 0.23271, and effective h 0.06268. Run 1408 is better on far-fluid and temperature quantities but worse on fluid, near-interface, vorticity, heat flux, and h. These are model-versus-dataset metrics, not new CFD validation or proof of physical causality.

## Learned sampling behavior

At P2, every query executes six groups by four sites: 24 fixed slots. Learned mass is much more concentrated than the slot count, averaging 4.305 nonzero sites per query in case 0273 and 4.412 in case 0653. Each query accesses about 44.58 and 44.33 distinct grid cells through bilinear corners, respectively. Across each complete case, the sites cover 154 cells and 184 of 192 fine-grid corner tokens.

The following frozen-model interventions used 1,024-query subsets. They measure learned reliance, not physical causality.

| Intervention | Case 0273 mean absolute field change | Case 0653 | Interpretation |
|---|---:|---:|---|
| Freeze sites to reference coordinates | 0.43584 | 0.53368 | Strong coordinate reliance |
| Replace learned within-group beta by neutral weights | 0.02236 | 0.03432 | Smaller but nonzero weight reliance |
| Safe module perturbation | 0.00534 | 0.00532 | Bounded local sensitivity |
| Remove P2 environmental contribution | 0.63301 | 0.69630 | P2 field materially uses environmental context |

Freezing sample coordinates increases case 0273 relative errors by +0.32086 heat flux, +0.89112 surface temperature, +0.64447 internal temperature, +0.24933 effective h, and +0.81816 outside temperature. The corresponding case 0653 increases are +0.29620, +0.86901, +0.69254, +0.23760, and +0.85558. Neutralizing beta produces much smaller but consistent changes. Removing only the final P2 environmental contribution changes the P2 field strongly while leaving P0/P1-derived port/interface quantities numerically unchanged; this is expected from the maintained coupling graph and should not be mistaken for irrelevance.

Actual accessed-cell visualizations:

![Run 1408 case 0273 executed sites and interpolation corners](../../diagnostics/generated/run1408_epoch500_evidence_20260920/figures/1408__0273__accessed_cells.png)

![Run 1408 case 0653 executed sites and interpolation corners](../../diagnostics/generated/run1408_epoch500_evidence_20260920/figures/1408__0653__accessed_cells.png)

## Executed work and measured cost

For one P2 evaluation with 8,192 queries, Run 1408 executes 196,608 site rows, 786,432 four-head content-dot rows, 196,608 geometry rows, and 786,432 bilinear corner loads. The materialized sampled bank is 105,381,888 bytes (100.5 MiB). The parent dense environmental rectangle contains 1,572,864 query-token rows, so the site-row count is one eighth as large; this is an operation-count observation, **not a speed claim**.

Synchronized CUDA measurements below use the same two anchor cases. Comparator figures come from the mature Run 1406/1407 comparison's measured architecture paths; checkpoint values do not change their dense execution shapes.

| Phase, mean of cases 0273/0653 | Run 1408 | Run 1406 | Run 1407 | Dense 1804 | Run 1404 |
|---|---:|---:|---:|---:|---:|
| Full forward | 51.86 ms | 38.90 ms | 38.21 ms | 33.98 ms | 25.38 ms |
| Preparation + one query | 36.22 ms | 28.66 ms | 26.26 ms | 21.85 ms | 22.45 ms |
| Prepared P2 | 21.16 ms | 12.03 ms | 14.08 ms | 13.02 ms | 6.39 ms |

The quadrature candidate is 35.7% slower than Run 1407 on full forward and 50.3% slower on prepared P2. Grid sampling, sample-bank materialization, and corner gathering outweigh the reduced site-row count on this implementation and hardware. No speedup is claimed.

Measured training-step medians on GPU 2, with 1,024 queries and batch size 48, are 0.748 s for M1 and 1.482 s for M12. Peak allocated memory is 3.09 GiB and 17.84 GiB; peak reserved memory is 3.87 GiB and 21.29 GiB. The mature comparison measured M12 at about 1.147 s for Run 1406, 1.146 s for Run 1407, and 2.106 s for Dense: Run 1408 is roughly 29% slower than either parent but about 30% faster than Dense in this training profile.

A separate real predicted-port backward/update completed in 0.544 s. Gradients and updated parameters were finite, and all five sampler parameter tensors changed (maximum absolute update 3.00e-4). This establishes gradient and optimizer connectivity, not convergence by itself.

## Verification

- Generic interpolation, exact-node identity, normalization, positive weights, gradient flow, receiver clamping, and Run 1408 plumbing tests pass.
- The dense-node reference identity and a real predicted-port backward/update pass.
- Trusted strict loading of the historical Run 1407 epoch-500 checkpoint passes.
- Core test suite: **637 passed**.
- Full repository suite with both case packages: **792 passed, 1 skipped, 4 failed**. All four failures are pre-existing path expectations for absent `HONF_Proj/diagnostics/evaluate_topology_quality.py` and `evaluate_retained_mass_pruning.py`; the tracked tools live under `tools/diagnostics`. They do not exercise Run 1408.
- `git diff --check` passes.

## Structured evidence

- Run manifest: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1408_20260920_133744_hypergraph_quadrature/run_manifest.json`
- Training metrics: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1408_20260920_133744_hypergraph_quadrature/metrics.csv`
- Epoch-500 execution/intervention evidence: `diagnostics/generated/run1408_epoch500_evidence_20260920/evidence.json`
- Sampling maps: `diagnostics/generated/run1408_epoch500_evidence_20260920/maps/`
- Matched 90-case evaluation: `diagnostics/generated/run1408_epoch500_matched_90case_20260920/`
- Reduced physical/tail/paired tables: `diagnostics/generated/run1408_epoch500_matched_90case_20260920/reduction/`
- Historical Run 1404 exact-500 metrics: `diagnostics/generated/interface_operator_study/run1406_epoch500_comparison_report/`

## Missing evidence and limits

- No new CFD truth, mesh-independence study, or out-of-distribution geometry validation was run.
- The 90-case comparison is a single-seed, exact-500 comparison; no second candidate or hyperparameter sweep was authorized.
- Mature parent best-through-500 weight files were unavailable, so the matched comparison uses exact epoch 500 rather than parent-selected checkpoints.
- Cost evidence covers two anchor cases on one RTX 6000 Ada GPU; it does not establish portability to other devices or kernels.
- Intervention evidence covers two anchor cases and 1,024-query subsets. It demonstrates frozen-model reliance, not causal physical meaning.
- The executor does not skip the low-mass slots, deduplicate repeated corners, or use a fused kernel. Learned mass concentration therefore does not reduce the fixed 24-site execution count.
- No 5000-epoch continuation was performed. Given the decisive measured latency regression and broad exact-500 physical deficit, longer training would not answer the failed computational claim.
