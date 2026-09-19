# Run 2000: module-hub routed pairwise HONF at epoch 500

**Assessment status: training in progress; final analysis awaits the user's instruction.** The user requested a pause after preparing the epoch-500 tools. This draft records completed implementation and pretraining checks; it does not assert an epoch-500 outcome. The [analysis handoff](HONF_Routing_Run2000_Epoch500_Handoff.md) contains the prepared, unexecuted final-analysis commands. The existing training process remains authorized to finish epoch 500.

## Scope and implementation

Goal 1 adds `routed_pairwise_honf` with `routing.strategy="module_hubs"`. The profile is [routing_module_hubs_context.json](../../src/config_core/forward/routing_module_hubs_context.json). Goals 2 and 3 are not implemented or launched by this task. The historical models, 1402/1403 decoder-context ablations, and WindFarm implementation are retained.

The backend inherits Dense's simultaneous fine MM/ME/EM preparation, environmental update, QM response network, and QE content/geometry attention. The shared coarse/local paths and ThermalChannel predicted-port P0/P1/P2 loop remain in use. Every phase prepares its current contextual states. Hubs contain routing descriptors and indices; they never replace fine field values.

Source memberships use ordinary sparsemax. Query routing solves `sum_k mu_k [z_k - tau]+ = 1`, returning density `d_k=[z_k-tau]+` and probability `alpha_k=mu_k*d_k`. The executable prior is `Pi_i=omega_i*sum_k A_ik*d_k`; it never divides by an almost-empty hub measure. Query normalization is not ordinary sparsemax applied to the same unscaled logits.

Positive query–hub and source–hub paths are joined and coalesced by receiver/source index before either QM messages or QE geometry/content scores are evaluated. Selected priors and prepared states retain gradients. Environmental quadrature enters through the source measure exactly once. Scalar projection and prior arithmetic use FP64; H-wide states remain in model dtype. Fine neural work is tiled, including its H-wide gathers, within non-reentrant training checkpoints. No occupancy cap, top-k truncation, hidden dense fallback, temperature tuning, or parent initialization was introduced.

The generic router receives geometry through a provider. The ThermalChannel adapter supplies its seven known boundary descriptors, length scale `4r`, and differentiable neutral extra resistance. An artificial barrier sampler test establishes implementation behavior only. **Barrier benefit: Evidence Missing.**

## Run and executed commands

Repository: `cosmos2w/ModularDT`, branch `agent/honf-core-next`. The implementation was committed before the formal launch. Later edits add diagnostic readers, plots, and tests; they do not change the running model.

Working directory for commands:

```text
/home/wanglz/Desktop/src/ModularDT/HONF_Proj
```

The only managed training launch was:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:Case_ThermalChannel/src OMP_NUM_THREADS=4 \
  /home/wanglz/miniconda3/envs/ModularDT/bin/python -u train.py \
  --config project://src/config_core/forward/routing_module_hubs_context.json \
  --workflow forward --device cuda:0 --epochs 500 \
  --run-id 2000 --run-name routed_module_hubs --yes \
  > diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/training.log 2>&1
```

The ordinary allocator created:

```text
Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_2000_20260915_225542_routed_module_hubs
```

Physical GPU 2 is the RTX 6000 Ada exposed as logical `cuda:0`. The launch supplied neither an initialization nor a resume checkpoint. The existing frozen Stage-A local module at `Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/best_model.pt` is the physical dependency, not a forward-model parent initialization. The existing 1402/1403 processes on GPUs 0/1 were left running.

Study artifacts reside at [run2000](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/). [Executed commands](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/executed_commands.md) records the actual checks and measurements. Checkpoints are retained in the allocator's run directory rather than copied into reports.

## Numerical and compatibility evidence

The focused routing suite passed 31 tests before launch and 33 after adding split-quadrature and empty-active-module coverage. It checks source-measure projection, tiny occupied measures, live composition gradients, a sparse dense oracle, uniform Dense equivalence, selected network-parameter gradients, source duplication with split quadrature, module permutation, padding, batch composition, query chunking, 2D/3D geometry, and real Stage-A P0/P1/P2 position/heat gradients. A separate compatibility command passed 124 tests across configuration, core/extension contracts, 3D forward, Regional/hierarchical/group paths, and WindFarm. Eight endpoint-reduction checks passed. Historical Legacy and Dense checkpoints also reconstructed strictly on CPU.

The float64 numerical checks are in [routing_numerical_checks.json](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/numerics/routing_numerical_checks.json). The [conditional fine-source fixture](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/numerics/conditional_fine_source_fixture.json) uses the actual QM and QE readers: selected-source JVPs are nonzero and an omitted source has exactly zero conditional direct effect. This fixture does not imply that a trained physical route omits that source. Full-model derivatives separately allow contextual preparation, common paths, and port feedback to change.

The packed prior matched its dense algebraic oracle to `1.11e-16` maximum absolute error, with 10 raw paths coalesced into four pairs. Query probability and measure-weighted density masses were exactly one in the fixture. Joint JVP/finite-difference maximum absolute error was `6.35e-12` at step `1e-5`; source quadrature and conditional hub splitting errors were each `5.55e-17`. At the explicit sparsemax support event, support changed from `{0,1}` to `{0}`, with one-sided first-coordinate derivatives `0.5` and `0`; the output gap at step `1e-6` was `5e-7`. This records continuity with a derivative jump rather than asserting a unique derivative at the kink.

Exactly two disposable real physical forward/backward/update checks ran before training, each with batch size 48 and 1,024 field queries per case. Neither allocated a managed run or saved weights.

| Check | Active modules/case | Step time (s) | Peak allocated (MiB) | Peak reserved (MiB) | Preclip norm | Update norm | Router gradient / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 1 | 4.13686 | 6,180.15 | 6,766 | 474.786 | 0.549905 | 0 / 0 |
| Large | 12 | 9.37194 | 33,862.72 | 37,304 | 610.702 | 0.564803 | 0.0436228 / 0.0645368 |

All checked losses, gradients, and updated parameters were finite. The singleton router's zero task gradient is expected because its only membership/probability is one. The disposable helper's first-step parameter-delta inventory excludes parameters that were lazy before the initial forward; its gradient/finite checks cover materialized parameters. The formal training observations capture parameters after materialization. First-batch norms are recorded at selected epochs; missing observations are not zeros and do not certify every batch.

## Endpoint accuracy, learning, and tails

Pending completed epoch-500 and saved-best-by-validation-field checkpoints. Both policies will use all 90 established development cases, 8,192 grid queries, predicted ports, checkpoint-owned normalization, and configured receiver chunk 128. Saved-best results will state their actual epoch separately. The existing split has been used for development and is not an untouched test set.

At the completed epoch-50 milestone, sampled validation field MSE was 0.232619. Recorded first-batch total preclip/update norms were 2.94970 / 0.131763; router norms were 0.0154323 / 0.0266564. Training remained finite with active updates. Mean P2 module fine pairs were 5.86602, while environmental fine pairs remained 192.0. These early observations do not establish final convergence, smooth monotonic loss, or an efficiency benefit.

The existing exact-500 scalar populations were checked for 90 unique cases and matching target energies/counts and strata. Their pooled fluid relative L2 values recompute to Legacy 1401 **0.1171479881**, Dense 1804 **0.0987410316**, and Regional 1806 **0.0966520664**. These are reference observations, not automatic rejection thresholds. Parent best-through-5000 weights will not be relabelled as best-through-500.

## Routing work, interventions, and derivatives

Pending final checkpoint measurements. Five anchors are fixed: 0273, 0653, 0283, 0298, and 0302. The ledger distinguishes active physical sources from padding, source fanout from true sources-per-hub occupancy, raw paths from deduplicated fine pairs, and source/query active hubs. P0's executed padded port receivers remain visible. The two-anchor omitted-source audit uses 32 fixed queries and reports environmental Dense-attention mass and module message norms separately from field error.

The six interventions are P2 uniform routing, P2 QM removal, P2 QE removal, P0 uniform routing, P1-only uniform routing, and P2 common coarse removal. Uniform routing uses the same trained response weights and contextual sources; it is not the separately trained Dense baseline. Prediction discrepancy and intervened-minus-normal ground-truth error remain separate.

Position checks use `0.01r` and `0.005r`. The design does not specify independent heat steps, so the helper fixes `0.01` and `0.005` normalized heat-input units before endpoint inspection and records their checkpoint-derived physical equivalents. Signed AD/FD results use mean temperature and the maintained upstream/downstream x-quartile pressure contrast; that contrast is not the full-grid inlet/outlet engineering KPI. Dataset-native units are not relabelled as SI units. Model AD/FD agreement is an implementation self-check, not a physical derivative certificate.

## Timing, memory, and profile

The pretraining actual execution profile completed on real anchor 0273 and generated `(M,E,Q)=(32,768,65536)` with all routing scopes captured. The authoritative [corrected profile](../../diagnostics/generated/interface_operator_study/dynamic_sparse_routing/run2000/timing/prelaunch_profile_corrected.json) reports median full-forward times of 126.176 ms and 2,383.091 ms respectively. These are untrained implementation measurements, not final checkpoint comparisons.

The first measurement retained unrelated prepared state in some memory baselines and used a synthetic domain inconsistent with the maintained scaling helper. The corrected measurement released those references, used the maintained domain/environment generation, and added complete routing scopes. Both actual measurement records are retained; only the corrected one supports these numbers. This changed measurement code rather than the scientific operator. CPU and GPU annotation durations must be read separately; inclusive scope sums are not end-to-end latency.

Final real-anchor, synthetic-shape, Dense/Legacy, chunk-128, active-only Dense fine-kernel, and memory results are pending. Parameter count, optimizer storage, checkpoint size, prepared/live allocation, activation peaks, reserved memory, and pair counts will be reported separately.

## Physical evidence and continuation

The established 16 independent pair/triple reference requests remain pending in `diagnostics/generated/interface_operator_study/stage3/reference_requests/pending_reference_requests.json`; no solver-development project was started. Learned route maps are not physical influence maps, and surrogate outputs are not new physical reference truth.

The continuation assessment awaits the complete endpoint and trajectory. Epoch 500 is an early assessment. Slower convergence alone is not an automatic rejection; the Dense/Regional historical maturity reversal is context, not a matched-budget baseline replacement. No continuation, new managed run, sweep, Goal 2, or Goal 3 has been launched. An exact unexecuted same-run continuation command will accompany the final recommendation if warranted.
