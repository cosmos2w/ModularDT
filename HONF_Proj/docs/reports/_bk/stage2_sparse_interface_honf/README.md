# HONF sparse-interface program: Stage 2 closeout and Stage 3 handoff

## Outcome

Stage 2 implemented `sparse_interface_honf` on the non-legacy `encode_case` / `prepare` / `read` family path introduced in Stage 1. The model uses layout-generated overlapping cubic B-spline supports, learned typed membership only inside those supports, and one nonlinear multi-entity state per occupied group. The same group states and keys/values supply autonomous physical-port contexts and continuous field reads. One layout and its module/environment incidence lists are reused through P0/P1/P2; environmental learned work is cached, while module messages, group states, and the common coarse state refresh after local-operator feedback.

The common eight-token coarse path, compact local correction, physical features and losses, frozen Stage-A disk operator, autonomous predicted ports, and one refinement are unchanged from the Stage-1 matches. The new model has no Run-1401 organizer, fake hyperedge arrays, learned rank/count estimator, residual deflation, K classifier, straight-through count gradient, or count/diversity regularizer. Historical models and strict checkpoint loading remain available, and the recommended profile remains `stage7_structured_context`.

Run 1802 completed exactly 500 epochs from scratch. At the matched endpoint it is a negative-but-informative research result: the 20-case mean fluid normalized-L2 error is `0.16156`, versus `0.10940` for Run 1401, `0.09352` for the dense adaptation, and `0.13744` for the latent adaptation. More decisively, its final sparse-group context is numerically unused: the 20-case query main-context fraction is `1.29e-25`, while coarse/local fractions are `0.898/0.102`. The candidate constructs and evaluates real groups, but its learned receiver read suppresses them and the common coarse/local paths carry the prediction. No post-hoc regularizer, restart, extension, or extra model was added.

## Implementation

- `src/honf_forward_core/interface_fields/supports.py` evaluates generic-dimensional tensor-product cubic B-splines, enumerates only the local `4**d` lattice candidates, coalesces physical-port footprints to module/group membership, retains boundary-overlapping centres, and builds flattened variable-K module/environment/receiver incidence lists.
- `src/honf_forward_core/interface_fields/group_operator.py` applies separate learned module/environment membership scorers, signed messages, `1+n` and fixed-volume non-singular pooling, the bounded `-expm1(-(4**d)*n)` occupancy envelope, one shared nonlinear group state, precomputed keys/values, and a stable null-normalized sparse reader.
- `src/honf_forward_core/interface_fields/core.py` registers the backend behind the shared family facade and preserves the matched coarse/local paths and field head.
- `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` builds the port-defined layout once, reuses its cache, and refreshes group/coarse states at the existing physical passes. Initial port prediction, provisional outside-temperature reads, final field decoding, and port consistency all dispatch to the selected backend.
- `src/config_core/forward/sparse_interface_honf_context.json` is the complete 500-epoch candidate profile with `support_spacing_factor=4.0`. The registry marks it as a candidate without changing the recommendation.
- Training and comparison reporting add optimization, physical port/interface, sparse incidence/routing, branch-magnitude, and measured-cost observations. These are ordinary diagnostics, not gates.

The sparse main path performs neighbourhood matching before its membership/message networks. Its neural work is proportional to retained module/environment incidences and at most 16 group incidences per 2-D receiver; it does not instantiate an all-module or query-by-K neural matrix. The inherited common coarse path intentionally uses its fixed latent attention, and the inherited local correction forms a cheap query/module distance tensor before running its MLP only on gathered local neighbours.

## Verification

| Check | Ordinary test/evidence | Result |
|---|---|---|
| Sparse geometry and gradients versus a tiny dense reference | `tests/test_sparse_supports.py::test_sparse_geometry_and_coordinate_gradients_match_tiny_dense_reference` | Passed; coordinate gradients matched at `rtol=2e-12`, `atol=2e-12`. |
| Learned prepare/read versus a dense reference | `tests/test_sparse_supports.py::test_learned_sparse_prepare_read_and_gradients_match_dense_reference` | Passed for memberships, typed pools, nonlinear state, keys/values, read context, module/receiver gradients, and every operator parameter; gradients used `rtol=2e-9`, `atol=2e-10`. |
| Joint module/port permutation and inactive padding | `tests/test_sparse_supports.py::test_module_permutation_padding_flat_batch_offsets_and_boundary_centres` and `Case_ThermalChannel/tests/test_interface_field_models.py::test_interface_fields_preserve_joint_module_permutation[sparse_interface_honf]` | Passed; end-to-end fields and restored ports used `rtol=3e-5`, `atol=3e-6`. |
| Environmental quadrature duplication | `tests/test_sparse_supports.py::test_environment_quadrature_duplication_preserves_learned_preparation_and_read` | Passed for environment pool, group state/keys/values, and read context at `rtol=2e-12`, `atol=2e-12`. |
| Query chunking | `tests/test_sparse_supports.py::test_environment_quadrature_duplication_and_receiver_chunking_are_invariant` and `Case_ThermalChannel/tests/test_interface_field_models.py::test_interface_fields_are_finite_chunk_independent_and_coordinate_differentiable[sparse_interface_honf]` | Passed; complete model used `rtol=2e-5`, `atol=2e-6`; receiver degree stayed at or below 16. |
| Support-boundary displacement | `tests/test_sparse_supports.py::test_support_transition_is_smooth_and_matches_directional_finite_difference` | Passed. At the zero-weight support exchange, autograd was `1.7337431913`, central difference was `1.7337431717`, and relative difference was `1.13e-8`; the test does not claim smoothness through unrelated physical discontinuities. |
| Cached topology/environment and refreshed states | `Case_ThermalChannel/tests/test_interface_field_models.py::test_sparse_environment_work_is_cached_while_group_states_refresh` | Passed across three preparations: one environment transform/pool construction and three distinct refreshed group states. |
| Strict checkpoint reconstruction | `Case_ThermalChannel/tests/test_interface_field_models.py::test_sparse_interface_checkpoint_reconstructs_strictly_with_bit_exact_output` | Passed with bit-exact field and port outputs. |
| Physical evaluation diagnostics | `Case_ThermalChannel/tests/test_sparse_evaluation_diagnostics.py` | Passed for flattened sparse serialization, shared group IDs, physical masked port/interface errors, debug arrays, and anchor plots. |

Focused command:

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q tests/test_sparse_supports.py tests/test_sparse_interface_config.py Case_ThermalChannel/tests/test_interface_field_models.py Case_ThermalChannel/tests/test_sparse_evaluation_diagnostics.py
```

Result: **23 passed in 3.83 seconds**.

The full ordinary suite command was:

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python -m pytest -q tests Case_ThermalChannel/tests
```

Result: **350 passed, 1 skipped**. The skip is the pre-existing inverse joint-training test whose optional local inverse artifacts are unavailable.

The actual packed-case GPU smoke command was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python Case_ThermalChannel/tests/check_global_modes.py \
  --config src/config_core/forward/sparse_interface_honf_context.json --device cuda:0 --points 32
```

It exercised fallback, teacher, predicted, and mixed port modes; all outputs were finite and the three coupled modes reported `interface_source=local_surrogate`.

An actual packed training case and trusted frozen Stage-A checkpoint were also used with a loss containing only predicted local internal/interface outputs. Backpropagation reached the sparse module-membership scorer, group value projection, receiver query, and port head with respective gradient norms `2.236e-6`, `4.258e-2`, `1.080e-3`, and `5.175e-1`; the local source was `local_surrogate` and the loss was finite (`1.1303`).

A 12-module/192-environment/1024-query execution probe observed K=64, 242 module incidences, 2,780 environment incidences, and 15,700 query/group read incidences (maximum degree 16). Forward hooks saw exactly those retained row counts at the membership/message/read networks, rather than the 65,536 entries of a dense Q-by-K read. Three preparations on one layout cache executed each environmental neural transform once and produced refreshed group states.

## Run 1802 execution record

| Managed run | Start | GPU | Epochs / updates | Trainable scalars | Train time | Validation time | Epoch work | Manifest wall | Peak CUDA allocation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Run_1802_20260905_170523_sparse_interface_honf` | from scratch | physical 0 | 500 / 6,500 | 3,778,064 | 8,399.80 s | 932.90 s | 9,332.69 s (2.592 h) | 9,413.42 s (2.615 h) | 29,582.44 MiB |

There were 13 real optimizer batches per epoch. The measured mean training-batch wall time, including forward, physical coupling, backward, clipping, optimizer update, and scheduled diagnostics, was `1.2923 s`; it is not an isolated kernel estimate. The run manifest is `completed`, contains epochs 1 through 500 and no later row, and records both `latest_model.pt` and `epoch_0500_model.pt`.

The allocator accepted proposed ID 1802 and created one from-scratch run on physical GPU 0. No occupied run was overwritten, no sweep or automatic continuation ran, and no support-matched control or additional model was launched.

### Optimization observations

| Epoch | Train total | Validation total | Validation field | Pre-clip total | Encoder | Group backend | Head | Local coupling | Clip scale | Update norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.71507 | 4.73394 | 1.94994 | 369.056 | 153.887 | 55.152 | 329.582 | 29.240 | 0.002710 | 0.52825 |
| 20 | 1.44905 | 1.30947 | 0.53709 | 28.058 | 8.252 | 3.338 | 26.366 | 3.579 | 0.035641 | 0.08368 |
| 500 | 0.04333 | 0.07691 | 0.03910 | 2.890 | 1.551 | 0.794 | 1.815 | 1.421 | 0.346004 | 0.06484 |

Gradient statistics use FP64 accumulation before clipping and were observed at epochs 1, 2, 5, 10, 20, and every 50 epochs. They are evidence, not numerical acceptance gates.

## Epoch-500 matched histories

| Model | Endpoint validation total | Best validation total through 500 | Endpoint validation field | Best validation field through 500 |
|---|---:|---:|---:|---:|
| Run 1401 legacy HONF | 0.05887 | 0.04474 (epoch 466) | 0.02360 | 0.01841 (epoch 466) |
| Dense pairwise adaptation, Run 1804 | **0.04518** | **0.03011** (epoch 493) | **0.01662** | **0.01339** (epoch 493) |
| Geometry latent adaptation, Run 1801 | 0.05693 | 0.05028 (epoch 490) | 0.02721 | 0.02466 (epoch 490) |
| Sparse interface HONF, Run 1802 | 0.07691 | 0.06514 (epoch 465) | 0.03910 | 0.03393 (epoch 465) |

The dense and latent entries are adaptations, not exact paper reproductions. Run 1804 is the completed Stage-1 dense endpoint; the preserved failed Run 1800/1803 allocations are not comparison entries.

## Matched 20-case physical comparison

The comparison reuses `docs/reports/stage1_interface_fields/development_cases.json`: cases 0273, 0653, 0277, 0291, 0298, 0680, 0281, 0654, 0643, 0644, 0276, 0667, 0274, 0293, 0657, 0665, 0282, 0292, 0688, and 0284. This is the established development holdout selected before model evaluation from module-count and spacing/clustering descriptors, not a pristine test set.

Equal-case means at the exact endpoints are below. The first three field columns are relative L2 in the common dataset-normalized field space. Remaining columns are MAE in each dataset-native physical quantity; they are not combined across unlike units.

| Model | Fluid | Near | Far | Fluid temperature MAE | Final port `T_env` MAE | Final port `h` MAE | Internal T MAE | Surface T MAE | Normal flux MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1401 @500 | 0.10940 | 0.10089 | 0.11911 | 0.5847 | 0.8246 | 0.3710 | 0.5537 | 0.6538 | 2.3941 |
| Dense adaptation @500 | **0.09352** | **0.08596** | **0.10077** | **0.4576** | **0.7050** | **0.3528** | 0.6061 | 0.7151 | 2.3984 |
| Latent adaptation @500 | 0.13744 | 0.10679 | 0.16568 | 0.5693 | 0.9541 | 0.3848 | **0.5232** | **0.5793** | **2.2062** |
| Sparse interface HONF @500 | 0.16156 | 0.12487 | 0.19639 | 0.7115 | 1.0707 | 0.3993 | 0.5500 | 0.6047 | 2.3981 |

| Case | Model | M | Fluid | Near | Far | Port `T_env` MAE | Port `h` MAE | Surface T MAE | Flux MAE | Sparse K | Mean/max members per group |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0273 | Run 1401 | 3 | 0.09294 | 0.07629 | 0.10362 | 0.8896 | 0.3173 | 0.4014 | 2.3240 | — | — |
| 0273 | Dense adaptation | 3 | **0.07654** | **0.06610** | **0.08348** | 0.7926 | 0.2904 | 0.4428 | 2.3657 | — | — |
| 0273 | Latent adaptation | 3 | 0.10273 | 0.07451 | 0.11949 | 0.7962 | 0.3496 | **0.3796** | **2.1303** | — | — |
| 0273 | Sparse HONF | 3 | 0.13845 | 0.09235 | 0.16461 | 0.9723 | **0.2817** | 0.3946 | 2.3369 | 34 | 2.03 / 3 |
| 0653 | Run 1401 | 5 | 0.10658 | 0.10275 | 0.11106 | 0.7782 | 0.4489 | 0.7155 | 3.1035 | — | — |
| 0653 | Dense adaptation | 5 | **0.09239** | **0.09518** | **0.08888** | **0.5871** | **0.3656** | 0.8166 | 3.0671 | — | — |
| 0653 | Latent adaptation | 5 | 0.13010 | 0.10358 | 0.15646 | 0.7279 | 0.3957 | **0.4273** | **2.8136** | — | — |
| 0653 | Sparse HONF | 5 | 0.14732 | 0.12067 | 0.17438 | 0.9109 | 0.4508 | 0.5020 | 2.9876 | 49 | 2.14 / 5 |

Physical error columns in the generated tables include value counts and MSE/RMSE/MAE/relative L2 for every fluid-field channel, internal temperature, surface temperature, normal flux, and provisional/final port `T_env` and `h_effective`. They remain separate from the normalized training objective.

## Sparse topology, branch use, and actual work

Counts are per case except for the final total. K is a geometry-generated occupied-support count, not a learned rank.

| Quantity | Min | Mean | Median | p95 | Max | 20-case total |
|---|---:|---:|---:|---:|---:|---:|
| Occupied groups K | 29 | 45.15 | 45 | 58 | 58 | 903 |
| Module/group incidence rows | 56 | 104.95 | 100.5 | 196.5 | 206 | 2,099 |
| Environment/group incidence rows, cached | 1,870 | 2,636.30 | 2,686.5 | 2,958.1 | 2,960 | 52,726 |
| Initial-port/group read rows | 6,627 | 9,128.15 | 8,998 | 11,672.4 | 11,794 | 182,563 |
| Final-query/group read rows | 80,762 | 112,851.85 | 114,832.5 | 127,086.1 | 127,259 | 2,257,037 |
| Case-mean unique modules/group | 1.217 | 2.258 | 2.158 | 3.451 | 3.679 | — |
| Case maximum unique modules/group | 2 | 4.65 | 4.5 | 9 | 9 | — |
| Case-mean field-read degree | 9.859 | 13.776 | 14.018 | 15.513 | 15.535 | — |
| Case-mean port-read degree | 8.629 | 11.886 | 11.716 | 15.198 | 15.357 | — |

The observed per-receiver maximum is 16, the exact `4**2` cubic-support bound. Across cases, the case-mean geometric occupancy is `0.1098`, envelope `0.4023`, learned module/environment scores `0.6282/0.5416`, and effective geometric-times-learned memberships `0.03066/0.01394`. Within every case, the port/query shared-group count equals K (903 when summed across cases), and every case passes the common-ID namespace check.

| Receiver | Main-group fraction | Coarse fraction | Local fraction | Mean nonnull group-read mass | Mean read degree |
|---|---:|---:|---:|---:|---:|
| Initial physical ports / P0 | 4.37e-24 | 0.75989 | 0.24011 | 1.61e-24 | 11.886 |
| Final field queries / P2 | 1.29e-25 | 0.89807 | 0.10193 | 7.01e-26 | 13.776 |

The topology is active and shared, but its receiver weights collapse: the trained candidate underuses the group branch at this budget. The implementation still refreshes group states at P0/P1/P2, uses P1 for provisional temperature reads, and builds environmental learned transforms/pools once per cached layout. The near-zero main fraction is therefore a learned branch-reliance result, not evidence that group construction was skipped.

Context-norm fractions are magnitude/reliance proxies, not causal ablations. Learned memberships are bounded routing coefficients, not physical correlations. Geometry generates K and the occupancy envelope; learned scores cannot change the group count.

One synchronized, no-warmup `predict_case` call per model/case (8,192 queries, including routing diagnostics) measured:

| Model | Mean wall s | Median wall s | p95 wall s | Mean queries/s | Mean incremental allocated MiB |
|---|---:|---:|---:|---:|---:|
| Run 1401 | 0.0547 | 0.0352 | 0.0580 | 222,021 | 363.12 |
| Dense adaptation | 0.1783 | 0.1775 | 0.1872 | 45,998 | 81.44 |
| Latent adaptation | 0.1481 | 0.1492 | 0.1545 | 55,400 | 54.91 |
| Sparse interface HONF | 0.2643 | 0.2629 | 0.2724 | 31,013 | 55.49 |

The cold first-model allocation makes these memory deltas useful only as observed end-to-end values, not a controlled memory ranking. A separate sparse-only run used five warmups and twenty synchronized repetitions:

| Case / phase | Median ms | p05–p95 ms | Incremental allocated MiB |
|---|---:|---:|---:|
| 0273 / full forward | 239.00 | 228.38–249.46 | 55.03 |
| 0273 / physical preparation + one query | 55.13 | 52.96–58.58 | 18.69 |
| 0273 / prepared decode, 8,192 queries | 181.58 | 175.36–184.76 | 52.84 |
| 0653 / full forward | 232.53 | 229.30–238.18 | 55.23 |
| 0653 / physical preparation + one query | 56.48 | 54.55–58.94 | 29.26 |
| 0653 / prepared decode, 8,192 queries | 174.81 | 170.90–181.16 | 52.84 |

Operation counts are retained rows actually processed, not masked dense dimensions or FLOP estimates. GPU time and operation count answer different questions. The synchronized matched-case cost table records one full `predict_case` call per model/case with no per-case warmups and includes routing diagnostics. The repeated sparse-only timing separates complete physical preparation from prepared decoding; Stage 3 retains the finer encoding/layout and training-step breakdowns.

## Artifacts and exact commands

- Run directory: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf`
- Endpoint checkpoint: `.../epoch_0500_model.pt`; complete optimization history: `.../metrics.csv`; managed record: `.../run_manifest.json` and `.../summary.json`.
- Comparison root: `Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case`
- Physical metrics: `tables/per_case_metrics.csv`, `tables/model_summary_metrics.csv`, and `tables/per_module_metrics.csv`.
- Sparse/branch evidence: `tables/interaction_case_metrics.csv` and `tables/interaction_summary_metrics.csv`. Legacy organizer evidence remains separately named in `tables/hypergraph_*`; the sparse K is never placed there.
- Measured evaluation cost: `tables/evaluation_cost_case_metrics.csv`, `tables/evaluation_cost_summary_metrics.csv`, `gpu0_phase_timing.csv`, and `gpu0_phase_timing.json`.
- Histories and figures: `tables/training_history_summary.csv`, `figures/training_histories.png`, and the matched anchor/support figures under `figures/interaction/`.
- Inspectable anchor and routing arrays: `debug_npz/`.

Run launch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py --config src/config_core/forward/sparse_interface_honf_context.json --run-id 1802 --device cuda:0 --yes
```

Training-history comparison:

```bash
PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/plot_stage1_training_histories.py \
  --series 'Run 1401 legacy HONF=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/metrics.csv' \
  --series 'Dense pairwise adaptation=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/metrics.csv' \
  --series 'Geometry latent adaptation=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/metrics.csv' \
  --series 'Sparse interface HONF=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf/metrics.csv' \
  --max-epoch 500 \
  --output-figure Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case/figures/training_histories.png \
  --output-table Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case/tables/training_history_summary.csv
```

Matched endpoint evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python evaluate.py \
  --config src/config_core/forward/sparse_interface_honf_context.json \
  --workflow compare \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1801_20260905_110056_geometry_latent_field_adaptation/epoch_0500_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf/epoch_0500_model.pt \
  --label 'Run 1401 legacy HONF @500' \
  --label 'Dense pairwise adaptation @500' \
  --label 'Geometry latent adaptation @500' \
  --label 'Sparse interface HONF @500' \
  --case-list docs/reports/stage1_interface_fields/development_cases.json \
  --split test --device cuda:0 --query-batch-size 32768 \
  --local-port-condition-mode predicted --return-routing-maps --save-debug-npz \
  --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case
```

Repeated synchronized sparse inference profiling:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/profile_stage2_sparse_inference.py \
  --checkpoint Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1802_20260905_170523_sparse_interface_honf/epoch_0500_model.pt \
  --case-id 0273 --case-id 0653 --device cuda:0 --warmups 5 --repetitions 20 \
  --output-dir Trained_Results/ThermalChannel/HONF_Forward_Runs/CompareModels/Stage2_Run1401_1804_1801_1802_Epoch500_20Case
```

The anchor figures for 0273 and 0653 contain physical predictions/errors for all models. Sparse figures additionally show every support footprint, geometric and learned memberships, unique-module degree, shared P0-port/P2-query group IDs, and query degree. The corresponding arrays are preserved in `debug_npz/`.

## Interpretation and limitations

At 500 epochs the dense adaptation is the strongest fluid-field candidate on this subset, while the latent adaptation is strongest on internal/surface/flux quantities. Sparse HONF is 72.8% worse than dense and 47.7% worse than Run 1401 in mean fluid normalized L2. It is not rescued by its topology plots: learned memberships remain finite and sparse execution is genuine, but the group-to-receiver mass collapses and the common coarse/local routes dominate. It is also the slowest of the four on these small real layouts. These observations motivate the Stage-3 conditional-influence and branch-intervention checks, not another automatic training run.

- This is one seed on an established 20-case development subset and does not establish final generalization.
- Run 1401 is compared at epoch 500. Its mature later checkpoint is context only, not a matched-budget competitor.
- The geometric cover is a prior, not a learned physical partition. Support-key changes and module insertion/deletion remain discrete even though support weights are smooth at the tested boundary transition.
- The common coarse route can carry long-range response and can bypass an underused fine group branch.
- Lower retained work does not guarantee lower GPU wall time on the existing small layouts.
- Receiver incidence lookup is currently recomputed for a new read; fixed layout/module/environment work is cached. Repeated receiver-neighbour caching is a Stage-3 performance refinement, not an accuracy change.
- The profile retains the Stage-1 `activation_checkpointing` runtime field for policy parity, but this compact sparse backend does not invoke checkpoint recomputation. Its measured training cost therefore contains no sparse checkpoint-recompute overhead.
- ThermalChannel validation here is two-dimensional. Formal large-module extrapolation and 3-D physics remain future work.
- Without the optional support-matched control, this result evaluates the complete sparse interface architecture; it does not prove that shared nonlinear groups uniquely outperform every sparse graph/kernel alternative.

## Stage 3 handoff

Stage 3 starts with evaluation only. Do not train, resume, sweep, or add a model automatically.

1. Build one synthesis matrix from Run 1401 @500, dense Run 1804 @500, latent Run 1801 @500, and sparse Run 1802 @500; extend the physical comparison across the existing 90-case development split and report physical strata.
2. Measure collective two-/three-module response against available physical references. Inspect the maintained solver first; use at most the plan's 16 solves/about two hours, or export requests and mark reference verification pending.
3. Run the three bounded branch interventions from the plan while recomputing downstream local responses: remove group context from ports, remove group field read, and clamp the coarse context. Distinguish causal effects from the norm proxies in this report.
4. Check model autograd against two-step central finite differences on four layouts, including an interior and a support transition. Model/self-FD agreement is not physical derivative validation.
5. Compare quadrature consistency for all four candidates, then run the five prescribed synthetic scaling shapes with the real support builder and label them execution-only.
6. On cases 0273 and 0653, measure phase-separated GPU-0 timing and allocated/reserved memory with about five warmups and twenty repetitions: encoding/layout, physical preparation/refinement, prepared decode, full forward, and a training step.
7. Produce matched figures for 0273, 0653, and one error-relevant larger-module case, then answer whether topology controls actual influence, whether the group branch participates, and which existing candidate—if any—merits a user-launched continuation.

An optional support-matched control or a 2.5K/5K continuation requires a new explicit user request. A longer HONF run against only 500-epoch baselines would be a development experiment, not a final fair superiority comparison.
