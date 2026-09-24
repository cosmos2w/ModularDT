# HONF 1503-v2: reversible edge coalescence

The generated evidence paths and one-time diagnostic scripts cited here are retained locally and intentionally excluded from GitHub; the numerical findings needed to read this report are recorded below.

## Experiment identity and evidence scope

This is the new `1503-v2 / reversible_edge_coalescence` experiment on `agent/honf-core-next`. Its scientific base is Run 1502's phase-local sparse-incidence model with final environmental sparsemax refinement. The archived Run 1503 adaptive-opening experiment remains separate. The normal allocator assigned distinct numeric Run 1504, named `1503_v2_reversible_edge_coalescence`, configured to start from the parent's seed and training settings without loading mature parent weights. The registered proposal bank remains K=12 with D=16 controls.

The 90-case `test` split is also used for sampled training validation, so full-grid results on it are development evidence. Single-run comparisons do not estimate training-seed variation. Group labels are permutation-ambiguous, and model-side routing does not identify physical causality.

An independent resolved-config comparison against the Run 1502 base profile plus its environmental-sparsemax overlay found differences only in the new architecture key, the three fixed fusion settings (64 ADMM steps, final eta 0.5, ramp 150 epochs), numeric run identity/name, physical device, and epoch budget. The seed, dataset, loss, optimizer, predicted-port policy, K/D, and source/query normalizers matched. A `train.py --dry-run` for the first 50 epochs resolved the 600/90 train/development split and proposed a new Run 1504 directory without allocating it.

The stored Run 1502 sampled validation field MSE provides same-epoch learning references, distinct from its mature checkpoint:

| Epoch | Run 1502 validation field MSE | Run 1502 validation temperature MSE |
|---:|---:|---:|
| 50 | 0.384350 | 0.211413 |
| 150 | 0.067827 | 0.033932 |
| 500 | 0.019280 | 0.013060 |

These are sampled validation metrics from the historical `metrics.csv`, not the 90-case full-grid relative-L2 comparison. The mature Run 1502 saved best-field checkpoint is epoch 4794, and is used below only for frozen implementation and cost references.

## Mathematical operator

Per prepared physical phase, the parent constructs twelve source incidences and controls. The coalescer solves a fixed 64-step convex-clustering ADMM problem on the parent query-access descriptors, with `eta(e)=0.5 min(e/150,1)`. It packs numerically equal access functions into case-local classes while retaining every fine physical source and each provisional controller parameter. The plan is reused across receiver chunks and rebuilt at the next physical phase.

For class `r` with multiplicity `m_r`, weighted sparsemax yields query mass `a_qr=m_r[L_qr-tau_q]_+` and per-constituent density `b_qr=a_qr/m_r`. The fine reader uses `b`, merged incidence `Abar_sr=sum_{k in C_r} A_sk`, and the source-resolved control moment `B_sr=sum_{k in C_r} A_sk h_k`. Thus `rho_qs=sum_r b_qr Abar_sr` and `n_qs=sum_r b_qr B_sr` match a virtual fine model with repeated fused access logits. This quotient equality does not assert that nonzero fusion preserves Run 1502 predictions: the fusion itself changes access functions and must be judged by trained physical outputs.

The environment retains one softmax over the unique supported source union, original contextualized K/V rows, and source-local value modulation from `sum_r B_jr`. The provisional `kappa_0` and literal K=12 finalization factors remain in place. Training uses the inherited rectangular fine reader; exact selected execution is measured separately.

## Prelaunch execution evidence

Focused CUDA:2 algebra tests passed for unequal-multiplicity weighted sparsemax against literal repeated logits and its gradient, source-incidence and heterogeneous-control moment conservation, zero-strength identity, footprint weights, ADMM residuals and gradients, and tolerance/iteration stability on seeded router-prepared descriptors. A separate integrated core suite passed strict state-dict loading, bitwise zero-strength parent field parity, a nonzero-strength backward/update, and unequal-chunk query-map aggregation. The first random nonzero-strength integration fixture retained twelve singleton classes, so a separate controlled physical-merge test forced one valid two-member class alongside singleton classes. That test passed on GPU 2: independent constituent sums verified module/environment Abar and B, the provisional kappa/pi remained unchanged, quotient mass matched literal repeated-logit sparsemax, and the compact QM+QE physical read and selected first gradients matched the virtual expanded parent reader within the asserted floating-point tolerances. The combined algebra, integration, and quotient gate passed 15 tests.

The maintained `run_epoch` path then performed one actual train-split predicted-port optimizer step at epoch 1 on physical `cuda:2`, without allocating a managed run or writing weights. Both batches traversed `p0_port`, `p1_refinement`, and `p2_field`, used AdamW with gradient clipping, and produced finite losses, all 151 gradient tensors, and nonzero finite updates:

| Actual train batch | Active modules | Queries/case | Update time | Peak allocated | Group-code gradient norm | Control-modulation gradient norm |
|---|---:|---:|---:|---:|---:|---:|
| 48 cases, mixed bucket | 2–4 | 1024 | 0.763 s | 8541 MiB | 0.00268 | 0.000874 |
| 15 cases, M=10 bucket | 10 | 1024 | 0.676 s | 6318 MiB | 0.00345 | 0.00132 |

The preflight artifacts are `diagnostics/generated/run1503_v2_preflight/real_update.json` and `real_update_m10.json`. These steps verify executable gradients and updates, not trained accuracy or physical derivative truth.

### Initial epoch-1 failure and same-run repair

The first managed Run 1504 launch failed during batch 9 of epoch 1, before any checkpoint or completed epoch. A same-seed, same-order replay found finite gradients and parameters after steps 1–7, followed by 2,847,646 nonfinite gradient values at step 8 and corrupted parameters after that update. The next provisional router preparation then reported an empty phase because its inputs and group codes were nonfinite. The initial managed traceback is preserved as `logs/initial_failure_run_manifest.json` in the Run 1504 directory.

The fusion layer was hardened at zero-footprint and inactive-group boundaries: squared footprint norms are floored before square root, wholly inactive descriptors and incidence are sanitized before masked calculations, the zero-variance case has a finite derivative, and zero-vector ADMM shrinkage avoids division by a tiny norm. Focused CPU regressions cover these cases; the combined fusion, integration, and controlled-quotient suite passed 17 tests on GPU 2. The same Run 1504 identity was restarted from its original seed in its existing managed directory because no checkpoint existed. All 13 epoch-1 training batches and both validation batches then completed; per-step telemetry found zero nonfinite gradients and zero nonfinite parameters after every update. The managed `metrics.csv` has exactly one epoch-1 row, and `latest_model.pt` is its first checkpoint. Epoch-1 train/validation total losses were 5.7737/4.2820 and field losses were 1.9393/1.8966. The replay log is `logs/repaired_e1_replay_retry.log`. This restart repairs the observed failure; later epochs still require monitoring for numerical stability.

On seeded synthetic cases passed through the real `SparseIncidenceGroupRouter.prepare`, 64 versus 128 ADMM steps gave identical R=12 class labels and at most 5.96e-8 fused-descriptor difference; halving the class tolerance from 1e-5 to 5e-6 did not change labels. The 64-step primal residuals were 4.06e-7/4.40e-7 and dual residuals 8.37e-8/8.94e-8. These checks do not imply convergence for every trained case. A controlled source-derived geometry surrogate crossed an R=11/R=12 merge boundary in a 4.66e-11-wide x bracket. Away from that boundary, model autodifferentiation matched centered differences at h=1e-3 and 5e-4 with absolute errors from 1.27e-14 to 4.28e-10. The reproducible command and inputs are in `tools/diagnostics/run_run1503_v2_fusion_sensitivity.py`, with results in `diagnostics/generated/run1503_v2_preflight/fusion_sensitivity.json`. This is solver self-consistency evidence, not a CFD sensitivity validation. A transition in a trained physical case remains to be observed.

A bounded frozen replay loaded the mature Run 1502 e4794 best-field checkpoint twice with identical weights: once in its original architecture and once through a temporary, strictly loaded 1503-v2 architecture configuration at final eta=0.5. On 1,024 fixed full-grid queries with predicted ports, cases 0273 (M=3) and 0686 (M=10) had field-vector relative changes of 0.01910 and 0.03559, respectively. Their maximum P2 descriptor displacements were 0.20063 and 0.30093, but both retained twelve singleton classes in all three phases. P2 primal/dual residuals were 2.78e-5/2.71e-6 and 2.18e-5/2.19e-6. The replay is reproducible with `tools/diagnostics/run_run1503_v2_frozen_replay.py`; its data are `diagnostics/generated/run1503_v2_preflight/frozen_parent_replay.json`. This isolates an operator change under frozen weights and shows neither trained fidelity nor useful coalescence at those two mature-parent geometries.

## Epoch-50 research review

Run 1504 completed exactly 50 epochs in the first managed stage after its repaired epoch-1 restart. Its `metrics.csv` has one row per epoch. Sampled validation field MSE was **0.419577** and temperature MSE **0.338252**, versus the stored Run 1502 same-epoch values 0.384350 and 0.211413. The candidate's field MSE is about 9.2% higher and its temperature MSE about 60% higher at this early stage; those sampled metrics are not a full-grid physical comparison. The candidate's fusion strength was only 1/6 of its final value. The trainer's per-epoch PyTorch peak allocated memory stayed near 24.2–24.3 GiB over the late part of this stage; `nvidia-smi` separately reported about 39,400 MiB of process GPU memory use. The latter is not a direct PyTorch reserved-memory measurement.

The exact `epoch_0050_model.pt` checkpoint was read on eight fixed development cases (0273, 0653, 0644, 0686, 0277, 0291, 0680, 0281), each with all 8,192 grid queries and predicted ports. The panel saved full field, internal temperature, interface, and port predictions; per-phase P0/P1/P2 formation and work ledgers; and spatial field/routing/Kq maps for 0273 and 0653 under `diagnostics/generated/run1503_v2/epoch_0050/`. Across these eight cases, final P2 mean Kq was 3.651 (range 3.149–4.120). All 24 observed case-phase plans had **singleton multiplicities**: K_case equalled K_proposal every time. P0, P1, and P2 each had seven cases with R=12 and one with R=11, but each R=11 arose from an unoccupied provisional group, not a fusion. Thus e50 shows query routing and a learnable field, but no genuine edge coalescence yet. The exact per-case field and formation records are in `epoch_review.json`. This candidate-only pass does not measure matched parent physical accuracy; its maps-on case wall times are diagnostic collection costs, not benchmark inference latency.

A separate maintained evaluator then read explicit Run 1502 and Run 1504 epoch-50 checkpoints on these same eight cases, Q=8,192, predicted ports, with the same masks and normalization. It saved 16 model-case rows, physical debug arrays, and checkpoint paths in `diagnostics/generated/run1503_v2/epoch_0050_paired_1502/`. Results below are relative L2 (lower is better), with pooled fluid metrics pooling squared errors/targets across cases and equal-case metrics averaging per-case errors:

| Eight-case physical metric | 1502 e50 | 1503-v2 e50 |
|---|---:|---:|
| Pooled normalized fluid field | 0.49261 | 0.53453 |
| Equal-case normalized fluid field | 0.48569 | 0.52800 |
| Pooled near-interface field | 0.42454 | 0.42738 |
| Pooled far-fluid field | 0.49934 | 0.57814 |
| Pooled physical u / v | 0.15014 / 0.54807 | 0.15450 / 0.57498 |
| Pooled physical pressure / vorticity | 0.51260 / 0.55488 | 0.50116 / 0.58105 |
| Pooled physical fluid temperature | 0.40500 | 0.53781 |
| Pooled internal / surface temperature | 0.20416 / 0.28365 | 0.19675 / 0.27537 |
| Pooled interface heat flux | 0.46749 | 0.46238 |
| Pooled final port temperature / effective h | 0.30476 / 0.14933 | 0.31189 / 0.10083 |

The new candidate had lower per-case normalized fluid error in 2/8 cases; its mean field error was higher, especially for fluid temperature and the far field. Some other physical metrics improved, so the field result should not be recast as universal physical regression. This is an eight-case development panel and one seed per model, not an independent generalization estimate.

The epoch-50 review supports continuing this same candidate to epoch 150: learning remains finite and improving, and the planned fusion schedule has not reached its endpoint. Absence of merges at eta=1/6 is recorded as an observed limitation; no smaller-K target or second fusion strength is introduced.
The solver nevertheless displaced access descriptors by roughly 0.05 in the representative P2 plans, so a change in routing or field error can occur before any discrete class merger. Because the models were separately trained from scratch, the paired e50 errors do not isolate the causal effect of this displacement.

## Epoch-150 review and scientific stop

The same managed Run 1504 continued from its e50 checkpoint to its exact e150 checkpoint without a new candidate allocation. Its manifest records 150 completed epochs and `metrics.csv` has exactly 150 rows. The fixed fusion schedule reached eta=0.5. Sampled validation field/temperature MSE at e150 were **0.095859/0.041127**, against Run 1502's stored same-epoch **0.067827/0.033932**. The best candidate sampled field MSE through e150 was 0.083747 at e149. These validation values fluctuate substantially over adjacent epochs, so the checkpoint comparison below is the more relevant physical readout.

For wider same-stage context, stored sampled e150 field/temperature MSE was 0.093755/0.032917 for Run 1501 and 0.065856/0.043538 for Dense Run 1804. These are historical run logs with distinct source snapshots and are not a controlled four-way causal comparison; no exact e150 Dense checkpoint was available for the matched full-grid panel.

The e150 fixed eight-case panel used the exact `epoch_0150_model.pt`, Q=8,192 and predicted ports. Its final P2 mean exact Kq was 3.779 (eight-case range and phase records in `diagnostics/generated/run1503_v2/epoch_0150/epoch_review.json`). A separate deterministic Q=1,024 formation pass covered **all 90 development cases and all three phases**. The prepared 64-step model had **zero non-singleton classes in all 270 case-phase plans**. Counts were P0 R12/R11 = 84/6, P1 = 73/17, and P2 = 74/16, and in every row R equalled the number of occupied proposals. All R11 outcomes therefore reflect empty proposal slots, not fusion. Mean sampled P2 Kq was 3.686 across the 90 cases (case range 3.252–4.271); with singleton classes, virtual constituent support equals coalesced Kq. The sweep record is `diagnostics/generated/run1503_v2/epoch_0150_formation_90_q1024/formation_sweep.json` with a per-phase CSV. The eight-case full-grid panel likewise showed only singleton classes. Its P2 primal residual reached 0.00364 in case 0281 (P0 reached 0.01003), much larger than the synthetic preflight residuals, so class decisions require the longer-solver check below.

The maintained ground-truth evaluator compared exact Run 1502 and Run 1504 e150 checkpoints on the same eight cases and complete Q=8,192 grids, with identical physical masks, normalization, and predicted ports. All values below are pooled physical or normalized relative L2, except the equal-case row, which averages eight individual normalized field errors:

| Eight-case e150 metric | 1502 | 1503-v2 |
|---|---:|---:|
| Pooled normalized fluid field | 0.20322 | 0.25109 |
| Equal-case normalized fluid field | 0.20104 | 0.24697 |
| Pooled near-interface / far-fluid field | 0.20422 / 0.22130 | 0.23232 / 0.27943 |
| Pooled physical u / v | 0.05178 / 0.19226 | 0.07547 / 0.26263 |
| Pooled physical pressure / vorticity | 0.22394 / 0.24962 | 0.26651 / 0.31073 |
| Pooled physical fluid temperature | 0.17729 | 0.18276 |
| Pooled internal / surface temperature | 0.07370 / 0.10621 | 0.09748 / 0.12795 |
| Pooled interface heat flux | 0.28133 | 0.28378 |
| Pooled final port temperature / effective h | 0.11658 / 0.04250 | 0.17136 / 0.05546 |

The candidate had higher normalized fluid relative L2 in **all 8/8 cases**, with per-case gaps 0.029–0.063. Equal-case physical fluid-temperature error was slightly lower for the candidate (0.20210 versus 0.21317), while pooled physical temperature was slightly higher; this weighting distinction prevents a claim of uniform degradation. The 16 case-model records and physical debug arrays are under `diagnostics/generated/run1503_v2/epoch_0150_paired_1502/`. This is matched same-epoch development evidence from one seed per architecture, not independent generalization or a causal isolation of fusion from training variation.

At e150, the final-strength 64-step implementation had not produced the intended case-conditioned merger on any surveyed physical phase, the paired field result regressed on every fixed case, and real-case solver residuals raised a numerical convergence concern. The run was **stopped at epoch 150**, before the authorized e500 ceiling; no 5000-epoch continuation was launched. The endpoint and the saved validation-best-field checkpoint are evaluated separately below. This stop is a research conclusion about this single fixed candidate, not an arbitrary scalar software gate.

The exact e150 trained checkpoint also passed a model-side gradient probe on predicted-port cases 0273 (M=3) and 0686 (M=10), using sixteen deterministic full-grid query points per case. First gradients for query coordinates, module centers, heat powers, local module properties, both source encoders, group codes, and module/environment control maps were finite and nonzero. Centered finite differences for the first query x and first module-center x at h=0.01 and 0.005 agreed with autodifferentiation to maximum absolute error 4.40e-5. The largest relative error was 4.52% for a small center derivative of magnitude 7.64e-4; reporting the absolute error avoids exaggerating that percentage. All perturbed P0/P1/P2 class partitions remained unchanged. The exact numerical record is `diagnostics/generated/run1503_v2/epoch_0150/design_gradients.json`, generated by `tools/diagnostics/run_run1503_v2_design_gradients.py`. It checks the trained surrogate's local derivative consistency at fixed source data, not CFD sensitivity or differentiability through an observed physical merge/split event.

### Trained-case solver sensitivity

A separate read-only diagnostic captured actual e150 prepared descriptors and footprint weights for cases 0273, 0686, 0281, and 0277. It replayed the same convex problem for 64, 128, 256, and 512 FP32 ADMM steps, plus half the numerical grouping tolerance at 64 and 512 steps. The saved checkpoint and trained forward remain the prescribed **64-step** model. At 64, all four cases had zero actual merges in each phase, consistent with the 90-case sweep. At 128, 256, and 512, both 0273 and 0686 stably merged provisional pair [6,7] in **P0, P1, and P2**, changing R from 12 to 11; those 128/256/512 partitions agreed exactly, and the 512-step partition survived halving the grouping tolerance. The 64-step partitions were unchanged by halving tolerance. Thus the production no-merge observation is real for its code path but is **not robust to solver iteration count**.

For 0273 P0, the primal residual fell from 2.83e-4 at 64 steps to 3.37e-7 at 128 steps; 0273 P2 fell from 4.38e-4 to 7.12e-7. For 0686 P2 it fell from 1.85e-5 to 2.03e-7. Case 0281 P0 needed 256 steps to reduce 0.01003 to 4.07e-7, although its partition remained singleton through 512; 0277 also stayed unchanged. Constituent-aligned fused descriptor differences between 64 and 128 were below 1.5e-4 on 0273, whereas raw packed-slot comparison would have exaggerated the change after indices shifted at a merge. The corrected artifact is `diagnostics/generated/run1503_v2/epoch_0150_solver_sensitivity_corrected/solver_sensitivity.json`, produced by `tools/diagnostics/run_run1503_v2_solver_sensitivity.py`. This supports a solver-convergence revision before any formal continuation; it is an evaluation-only alternative and cannot retroactively turn the recorded 64-step training run into a successful formation experiment.

## Full 90-case development comparison at the stop

The maintained evaluator read exact Run 1502 e150, candidate endpoint e150, and candidate saved validation-best-field e149 checkpoints. It evaluated each over the same 90 development cases, full Q=8,192 grids, identical physical masks/normalization, and predicted ports. The candidate best-field selector chose e149 based on sampled validation field MSE; this is a checkpoint-policy sensitivity check, not an extra training run. Results are pooled relative L2 except the equal-case field row, which averages 90 per-case errors:

| 90-case metric | 1502 e150 | Candidate e150 endpoint | Candidate e149 best field |
|---|---:|---:|---:|
| Pooled normalized fluid field | 0.20215 | 0.25064 | 0.21635 |
| Equal-case normalized fluid field | 0.20020 | 0.24629 | 0.21339 |
| Pooled near-interface field | 0.20228 | 0.23326 | 0.22330 |
| Pooled far-fluid field | 0.22036 | 0.28340 | 0.22301 |
| Pooled physical u | 0.05844 | 0.07484 | 0.06223 |
| Pooled physical v | 0.18413 | 0.26000 | 0.22114 |
| Pooled physical pressure | 0.20925 | 0.25348 | 0.21754 |
| Pooled physical vorticity | 0.25489 | 0.31772 | 0.28174 |
| Pooled physical fluid temperature | 0.16794 | 0.17269 | 0.14463 |
| Pooled internal / surface temperature | 0.07181 / 0.10541 | 0.08182 / 0.10752 | 0.07047 / 0.10242 |
| Pooled interface heat flux | 0.29055 | 0.29166 | 0.28540 |
| Pooled final port temperature / effective h | 0.11693 / 0.05510 | 0.15212 / 0.06820 | 0.12429 / 0.06231 |

The endpoint was worse than the parent on normalized fluid field in **89/90** cases, with median paired per-case gap +0.04871 and 95th-percentile gap +0.07008. Its worst gap was +0.07889 on case 0675; the only field win was case 0289 by 0.00393. The saved e149 best-field checkpoint was closer but still worse in **71/90** cases, median gap +0.00979 and 95th-percentile gap +0.03750; its worst gap was +0.05130 on case 0643. By active-module count M=3/5/7/10, endpoint mean normalized field errors were 0.22529/0.23790/0.26072/0.27122, versus parent 0.18979/0.20226/0.20148/0.21196. Best-field means were 0.20124/0.20740/0.21805/0.23583. The best checkpoint improves temperature and a few thermal-interface metrics, so the research result is a field and formation failure, not uniform degradation of every physical quantity. Pooled endpoint field error is 16% above its own e149 best-field error; treating the saved best as the e150 endpoint would hide this sensitivity. The paired case tables and debug arrays are under `diagnostics/generated/run1503_v2/full90_e150_paired_1502/` and `diagnostics/generated/run1503_v2/full90_bestfield_e149_paired_1502/`.

The separate e149 best-field formation sweep found one actual two-member class in case 0676, in P1 and P2 only: ten singleton classes and one multiplicity-two class, R=11 from twelve occupied proposals. All other case-phase plans were singleton (P0 R12/R11=86/4, P1=76/14, P2=76/14; the remaining R11 plans had inactive proposal slots). Across 90 cases, mean sampled e149 P2 Kq was 3.705. This **one-case merge** is fragile at the prescribed 64 steps: halving the numerical grouping tolerance removes it in P1/P2, while 128/256/512 steps retain pair [6,7] and a 512-step half-tolerance check restores it. In case 0676 P0, 64 steps leave R=12 but 128/256/512 stably merge a different pair [6,9]. This read-only solver check is in `diagnostics/generated/run1503_v2/epoch_0149_solver_sensitivity_case0676/solver_sensitivity.json`. The e149 formation distribution is in `diagnostics/generated/run1503_v2/epoch_0149_formation_90_q1024/formation_sweep.json`. Thus the saved best checkpoint exhibits a literal merge in the production path, but neither stable widespread case-conditioned coalescence nor robust numerical class decisions.

On that actual e149/P2 merged case, an independent source-bank check found class index `[6,7]→6`, Kproposal=12 and R=11. Across seven active modules, positive incidence entries fell from 37 to 35 and mean source degree from 5.286 to 5.000; among 192 environment rows, both remained 421 entries and mean degree 2.193. Source row sums stayed within 1.19e-7 of one. Direct constituent scatter sums matched both prepared `Abar` and source-resolved `B` with zero displayed FP32 difference; quotient-vs-virtual repeated-router rho and n differed by at most 1.19e-7 over sixteen deterministic queries. The reproducible diagnostic and result are `tools/diagnostics/run_run1503_v2_actual_merge_structure.py` and `diagnostics/generated/run1503_v2/epoch_0149_actual_merge_0676_structure.json`. It verifies moment preservation in a trained true-merge case; it does not establish physical accuracy relative to Run 1502.

A trained physical-coordinate sweep perturbed active module slot 0 by 21 x offsets from −0.25 to +0.25 physical units on exact e150 cases 0273 and 0686, with sixteen fixed full-grid query points and all other inputs held fixed. Every geometry remained feasible (minimum module surface clearances 0.716 and 0.478 radii, respectively). All 42 evaluations retained twelve singleton groups in P0/P1/P2. The reproducible record is `diagnostics/generated/run1503_v2/epoch_0150_physical_coordinate_sweep_m0_x.json`, from `tools/diagnostics/run_run1503_v2_physical_coordinate_sweep.py`. No trained 64-step physical merge/split boundary was observed in this bounded sweep, so the local AD/finite-difference checks above do not validate derivatives through such a transition. The earlier synthetic surrogate boundary is a separate numerical check.

## Corrected parent executor replay

The optional `support_blocks` dispatch previously executed its block reader and then fell through to the partial reader, overwriting the first result. The repair makes the alternatives exclusive and moves the fixed rectangular module policy ahead of the expensive support test. Executed-call tests invoke real numerical readers and check parity. This defect affects interpretation of earlier optional block timings, not the historical default rectangular accuracy table.

The corrected mature Run 1502 replay used its static validation-best-field checkpoint at epoch 4794, cases 0273 and 0653, all 8,192 grid queries, physical GPU 2 (RTX 6000 Ada), PyTorch 2.6.0+cu124, maps off during timing, two warmups, three synchronized repetitions, and explicit inner receiver chunks. The inherited Torch selected path is a reference that performs one fine read per supported pair but gathers K/V; it is distinct from the new fused CSR prototype. Values below are median wall milliseconds. Full GPU forward, prepared P2 decode, and complete application are distinct scopes and should not be added together.

| Inner chunk | Case | Reader | Full GPU forward | Prepared P2 decode | Application |
|---:|---|---|---:|---:|---:|
| 128 | 0273 | Rectangular | 225.04 | 166.20 | 261.98 |
| 128 | 0273 | Corrected support blocks | 1451.95 | 1362.91 | 1490.29 |
| 128 | 0273 | Torch selected pairs | 251.97 | 189.34 | 295.45 |
| 128 | 0653 | Rectangular | 230.66 | 152.32 | 249.40 |
| 128 | 0653 | Corrected support blocks | 1480.35 | 1377.18 | 1474.38 |
| 128 | 0653 | Torch selected pairs | 267.60 | 194.62 | 297.69 |
| 2048 | 0273 | Rectangular | 48.41 | 14.01 | 54.49 |
| 2048 | 0273 | Corrected support blocks | 236.28 | 180.05 | 250.57 |
| 2048 | 0273 | Torch selected pairs | 111.99 | 72.50 | 124.58 |
| 2048 | 0653 | Rectangular | 49.88 | 14.71 | 53.14 |
| 2048 | 0653 | Corrected support blocks | 264.94 | 172.78 | 274.09 |
| 2048 | 0653 | Torch selected pairs | 110.13 | 67.26 | 114.96 |

The selected Torch reader reduced measured QE geometry rows from about 1.87 million to 0.78–0.84 million in the instrumented complete forward but remained slower. The corrected block reader also remained slower and made many more geometry MLP calls. Larger evaluation-only chunks substantially improved the rectangular reference too; that common chunking effect must not be attributed to coalescence. Exact per-case timing, CUDA-event, memory, parity, and invocation records are in `diagnostics/generated/run1503_executor_parent1502/parent1502_chunk128.json` and `parent1502_chunk2048.json`.

### Fused CSR prototype on the static parent

The optional fused CSR reader extends the existing selected-source kernel with the live pair/head score multiplier. The new kernel computes one softmax over each query's complete supported source union and has first-order gradients for query, key, value, geometry bias, prior, and score multiplier. A small full-core CUDA parity test also checked output and first gradients against rectangular execution. It remains an evaluation-only opt-in; the rectangular policy is unchanged.

At inner chunk 2048 on the same mature 1502 checkpoint and anchors, untimed invocation probes observed six fused QE calls, six geometry MLP calls, and 841,785 / 779,255 geometry rows for cases 0273 / 0653, versus 1,867,776 rectangular rows. Max output differences were 5.01e-6 / 6.20e-6. Two warmups and three synchronized repetitions gave:

| Case | Reader | Full GPU forward | Prepared P2 decode | Application | Application peak incremental allocation |
|---|---|---:|---:|---:|---:|
| 0273 | Rectangular | 50.60 ms | 14.07 ms | 54.05 ms | 457 MiB |
| 0273 | Fused CSR | 50.21 ms | 14.33 ms | 53.67 ms | 289 MiB |
| 0653 | Rectangular | 49.43 ms | 14.06 ms | 51.23 ms | 457 MiB |
| 0653 | Fused CSR | 50.55 ms | 14.49 ms | 52.81 ms | 247 MiB |

The prototype reduces allocation but does not establish a useful full-application speedup over the optimized rectangular reference on this panel. The complete per-scope samples are in `diagnostics/generated/run1503_executor_parent1502/parent1502_chunk2048_fused.json`.

A matched evaluation-only chunk-2048 Dense Run 1804 replay on the same two cases gave complete application medians 41.15 / 38.55 ms and full-GPU-forward medians 37.86 / 35.81 ms, faster than either Run 1502 mode at this shape. Prepared P2 decode was 13.84 / 13.44 ms. The Dense probe also executed six QE reader calls and 1,867,776 geometry rows per case. Its record is `diagnostics/generated/run1503_executor_parent1502/dense1804_chunk2048.json`. This comparison concerns frozen executor cost; it is not a same-epoch training-accuracy comparison.

### Fused CSR prototype on the epoch-50 candidate

A controlled true-merge R<12 full-core CUDA test passed selected-versus-rectangular output and first-gradient parity, exercising both the source-resolved QM moment and fused QE selected reader. The default candidate training policy remains rectangular. The actual epoch-50 checkpoint had only singleton packed classes on the fixed panel, so its timed replay assesses the optional exact selected executor at that checkpoint, not the speed of physical coalescence.

Using the exact `epoch_0050_model.pt`, Q=8,192, inner chunk 2,048, the same GPU 2, FP32, two warmups and three synchronized repetitions, the median milliseconds were:

| Case | Reader | Full GPU forward | Prepared P2 decode | Complete application | Full-forward incremental allocation |
|---|---|---:|---:|---:|---:|
| 0273 | Rectangular | 110.83 | 12.16 | 112.89 | 471 MiB |
| 0273 | Fused CSR | 109.29 | 16.17 | 111.94 | 364 MiB |
| 0653 | Rectangular | 103.36 | 12.09 | 107.89 | 471 MiB |
| 0653 | Fused CSR | 107.36 | 15.58 | 107.12 | 311 MiB |

Untimed invocation probes observed six QE calls per mode. Fused dispatch executed its selected QE path and reduced actual geometry MLP rows from 1,867,776 per case to 1,014,861 / 927,183 on cases 0273 / 0653. Maximum output differences were 1.27e-5 / 1.44e-5. The prepared P2 decode became about 29–33% slower, while complete application time was effectively unchanged at these repetitions; fewer rows are not an acceleration claim. The detailed invocation, CUDA-event, memory, and sample record is `diagnostics/generated/run1503_executor_parent1502/candidate1504_e50_chunk2048_fused.json`. The parent and Dense cost comparisons above use different checkpoints and should be read as implementation context, not equal-stage trained accuracy comparisons.

### Exact execution at the scientific stop and saved best-field checkpoint

The same benchmark protocol was repeated on the exact e150 endpoint and independently selected e149 best-field checkpoint, both with Q=8,192, chunk 2,048, GPU 2, two warmups, three synchronized repetitions, maps off for timing, and untimed executed-call probes. Values are median wall milliseconds; full GPU forward, prepared P2 decode, and complete application are separate scopes.

| Checkpoint | Case | Reader | Full GPU forward | Prepared P2 decode | Application | Full-forward incremental allocation |
|---|---|---|---:|---:|---:|---:|
| e150 endpoint | 0273 | Rectangular | 108.15 | 12.16 | 115.50 | 449 MiB |
| e150 endpoint | 0273 | Fused CSR | 105.43 | 16.00 | 114.07 | 295 MiB |
| e150 endpoint | 0653 | Rectangular | 105.13 | 12.65 | 109.55 | 449 MiB |
| e150 endpoint | 0653 | Fused CSR | 109.82 | 15.38 | 111.65 | 296 MiB |
| e149 saved best field | 0273 | Rectangular | 103.46 | 11.88 | 110.52 | 449 MiB |
| e149 saved best field | 0273 | Fused CSR | 110.56 | 15.70 | 108.76 | 295 MiB |
| e149 saved best field | 0653 | Rectangular | 101.19 | 11.88 | 109.09 | 449 MiB |
| e149 saved best field | 0653 | Fused CSR | 107.47 | 15.43 | 113.75 | 292 MiB |

Both modes executed six QE calls per complete forward. The e150 endpoint's actual geometry MLP rows fell from 1,867,776 rectangular to 907,576 / 961,568 fused for cases 0273 / 0653; the e149 best-field values were 909,149 / 947,492 fused. Fused-vs-rectangular maximum field differences were at most 1.26e-5 at e150 and 5.48e-6 at e149. The optional executor consistently saved incremental allocation but gave mixed full-application timing, with P2 decode slower in every row. There is no defensible application-speed claim or evidence that fewer prepared groups caused these savings: both evaluated checkpoints retained singleton groups in these cases, and the measured row skipping comes from exact source support. Full per-repetition wall/CUDA-event timing, source-pair ledgers, and actual invocation counts are in `diagnostics/generated/run1503_executor_parent1502/candidate1504_e150_endpoint_best_chunk2048_fused.json`.

The lone actual-merge case at saved e149, **0676**, was benchmarked under the same protocol. It had R=11 from a two-member class in P2, yet fused CSR complete application was **113.69 ms versus 107.36 ms rectangular**; full GPU forward was 106.99 versus 99.90 ms and prepared P2 decode 14.90 versus 12.13 ms. The fused reader reduced complete-forward geometry rows from 1,867,776 to 819,246 and incremental allocation from 449 to 278 MiB, with maximum output difference 7.63e-6. This directly measures an actual merged model case and shows that reduced logical groups and skipped physical rows did not translate into acceleration. The record is `diagnostics/generated/run1503_executor_parent1502/candidate1504_e149_actual_merge_0676_chunk2048_fused.json`.

The final exact-execution replay covered **all 90 endpoint cases** at Q=8,192 and chunk 2,048 under the same two-warmup, three-repetition synchronized protocol. Means below average each case's median wall time; they are not medians of a pooled timing stream.

| Scope across 90 cases | Rectangular mean | Fused CSR mean | Cases where fused was faster |
|---|---:|---:|---:|
| Full GPU forward | 104.22 ms | 109.22 ms | 11/90 |
| Prepared P2 decode | 12.09 ms | 15.24 ms | 1/90 |
| Complete application | 109.63 ms | 112.96 ms | 25/90 |

Both paths made six geometry-MLP calls per complete forward. Fused CSR reduced actual executed geometry rows from 168,099,840 to 79,110,182 across the 90 cases (mean 1,867,776 to 879,002 per case) and mean full-forward incremental allocation from 449 to 291 MiB. The median per-case fused/rectangular application-time ratio was 1.032, so the selected reader was about **3% slower** on this full development population despite roughly 53% fewer geometry rows. Max absolute prediction difference across all cases was 6.01e-5 (case 0660); the mean of case-wise mean absolute differences was 1.86e-7. This is FP32 numerical agreement, not bitwise identity. The detailed timing and actual-call evidence are in `diagnostics/generated/run1503_executor_parent1502/candidate1504_e150_full90_chunk2048_fused.json`. The default rectangular reader remains the measured faster choice at this shape.

## Recommendation and limits

**Do not launch a 5000-epoch continuation of this Run 1504 checkpoint.** The prescribed 64-step solver yielded no endpoint merges across 270 surveyed physical phases; the only saved-best merge was tolerance-sensitive, and longer-solver replays changed several real-case partitions. The exact e150 endpoint also trailed same-epoch Run 1502 on pooled field, near/far fields, most per-field physical metrics, and 89/90 individual normalized field cases. The saved e149 best field improved temperature and some thermal-interface measures but still trailed on pooled fluid field and 71/90 cases. The exact selected executor saved rows and memory without reducing application latency on the full population or the one actual merged case. A formal continuation would spend substantially more training compute without a numerically reliable formation mechanism or a competitive same-stage field trajectory.

For a future *new, separately labeled* experiment, first make the convex solve's class decisions stable at a measured residual tolerance on actual prepared phases, preserve the existing quotient/moment parity tests, then reassess same-stage physical fidelity and execution. Changing the ADMM iteration count for the existing e150 checkpoint would change its forward operator; the longer-step replays in this report are diagnostics, not a repaired continuation of the trained model. Keep rectangular execution as the default until a synchronized end-to-end benchmark demonstrates a real application win. No additional seed, lambda sweep, or 5000-epoch run was launched here.

All physical comparisons use the 90-case development split also used for sampled validation, with one seed per historical model and source-snapshot differences among older runs. The eight-case e50/e150 panels are diagnostic subsets; the full90 results are stronger within this development population but do not establish independent generalization. Model-side coordinate derivatives do not verify CFD sensitivities; the bounded trained coordinate sweep found no merge/split transition. Synthetic boundary and longer-solver replays should not be described as trained physical transition evidence.

The final focused GPU 2 regression suite passed **35/35 tests** across fusion, full coalesced core and quotient, actual fused-executor gradients, corrected dispatch, and Triton QE. The scripts passed Ruff and Python syntax checks, and `git diff --check` was clean. Historical Run 1501/1502/failed 1503 directories and checkpoints were read-only; the single managed 1503-v2 candidate is Run 1504 and its manifest ends at epoch 150. Existing strict checkpoint loading and security behavior were retained.
